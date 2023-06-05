import os
import re
import fitz
import json
import requests
import numpy as np
from PIL import Image
import torch
import pytesseract
from skimage import io as skio
from scipy.ndimage import interpolation as inter
import cv2
import logging
from transformers import LayoutLMv2Processor, LayoutLMv2ImageProcessor, LayoutLMv2ForSequenceClassification, LayoutLMv2Tokenizer
import logging

logger = logging.getLogger('ad_logger')

UPLOAD_FOLDER = '/flask_app/files/'
# ---- THIS DEPENDS ON THE MODEL ----
PROCESSOR_PATH = "microsoft/layoutlmv2-base-uncased"
MODEL_PATH = "DataIntelligenceTeam/BoLDocClassifier"
lang = 'eng'

base_url = 'https://ad5m0-test.wenda.cloud:{}/activedocuments/{}/{}'
headers = {'Accept': 'application/json'}

urlport4doctype = {
    'invoices': 6013,
    'LOC': 6012,
    'BOL': 6014
}

# -----------------------------------

label2idx = {
    'invoices': 0,
    'LOC': 1,
    'BOL': 2,
    'other': 3
}
id2label = {
    0: 'invoices',
    1: 'LOC',
    2: 'BOL',
}

# -----------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LayoutLMv2ForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
feature_extractor = LayoutLMv2ImageProcessor()
tokenizer = LayoutLMv2Tokenizer.from_pretrained(PROCESSOR_PATH)
processor = LayoutLMv2Processor(feature_extractor, tokenizer)


#endpoint= 'factura/uploader'
def upload_file(filepath, port, doc_type, endpoint, data):
    files = {'file': open(filepath, 'rb')}
    url = base_url.format(port, doc_type, endpoint)
    print(data)
    response = requests.post(url, headers=headers, files=files, data=data)
    json_data = response.json()

    # Extract task ID
    task_id = json_data['task_id']
    return task_id


def elab(filePath, ocr):
    logger.info('elab start')
    results = process_PDF(filePath, ocr)
    # Convert the dictionary to a JSON string
    json_data = json.dumps(results)
    logger.info('elab ends')
    return json_data, results


def process_page(filePath):
    logger.info('process_page start')
    image = Image.open(filePath).convert("RGB")
    page_class = infer(image)
    logger.info('process_page ends')
    return page_class


def infer(image):
    logger.info('infer start')
    encoded_inputs = processor(image, return_tensors="pt",truncation=True).to(device)
    logger.info('processor encoding OK')
    outputs = model(**encoded_inputs)
    logger.info('forward pass OK')
    preds = torch.softmax(outputs.logits, dim=1).tolist()[0]
    pred_labels = {label:pred for label, pred in zip(label2idx.keys(), preds)}
    pred_labels_ = max(pred_labels.values())
    keys = [k for k, v in pred_labels.items() if v ==pred_labels_][0]
    logger.info('infer end')
    return keys


def cleanup(pattern):
    for f in os.listdir(UPLOAD_FOLDER):
        if re.search(pattern, os.path.join(UPLOAD_FOLDER, f)):
            os.remove(os.path.join(UPLOAD_FOLDER, f))


def checkRotation(filePath):
    im = skio.imread(filePath)
    try:
        newdata = pytesseract.image_to_osd(im, nice=1)
        rotation = re.search('(?<=Rotate: )\d+', newdata).group(0)
    except:
        # Exception might happen with blank pages (tesseract not detecting anything)
        # so to mark it we set rotation = -1
        rotation = -1
    return rotation


def remove_borders(img):
    result = img.copy()

    try:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) # convert to grayscale
    except:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        gray = result[:, :, 0]
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 5)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 5)
    return result



# correct the skewness of images
def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    # Convert the PIL Image object to a numpy array
    image = np.asarray(image.convert('L'), dtype=np.uint8)

    # Apply thresholding
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)
    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)
    return best_angle, corrected


def process_PDF(filePath, ocr):
    logger.info('process_PDF start')
    # we unpack the PDF into multiple images
    doc = fitz.open(filePath)

    result = {}
    # for i in range(0, doc.page_count):
    # CAREFUL - in this case we consider ONLY the first page, assuming it contains the only relevant document.
    for i in range(0, doc.page_count):
        page = doc.load_page(i)     # number of page
        zoom = 2                    # zoom factor
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix = mat, dpi = 300)
        if filePath[-4:] == ".pdf":
            imgOutput = filePath.replace(".pdf", "_{}.png".format(i))
        elif filePath[-4:] == ".PDF":
            imgOutput = filePath.replace(".PDF", "_{}.png".format(i))
        pix.save(imgOutput)
        rotation = checkRotation(imgOutput)
        print(rotation)
        if rotation == -1:
            # img is blank: delete it?
            os.remove(imgOutput)
            result[imgOutput] = 'other'
            continue
        elif rotation != 0:
            im = Image.open(imgOutput)
            rotated = im.rotate(-(int(rotation)), expand=True)
            # angle, skewed_image = correct_skew(rotated)
            # print(angle)
            # out = remove_borders(skewed_image)
            out = rotated
            out.save(imgOutput)
            # cv2.imwrite(imgOutput, out)
        else:
            # im = Image.open(imgOutput)
            # angle, skewed_image = correct_skew(im)
            # print(angle)
            # out = remove_borders(skewed_image)
            # out = skewed_image
            # cv2.imwrite(imgOutput, out)
            pass
        # out.save(imgOutput)
        # each image goes through the model
        page_class = process_page(imgOutput)
        result[imgOutput] = page_class

    # clean up function to delete local files - both the original pdf
    # (that was previously uploaded to S3) and the newly created images
    logger.info('process_PDF end')
    return result


def send_to_elaboration(filePath, classification_result, webhook, pathfile, localpath):
    data = {
        'user': '',
        'save': '',
        'output': 'excel',
        'ocr': '',
        'webhook': webhook,
        'pathfile': pathfile,
        'localpath': localpath
    }

    result = dict()
    i = 0
    for page in classification_result:
        page_category = classification_result[page]
        if page_category in urlport4doctype.keys():
            port = urlport4doctype[page_category]
            data['pagnr'] = str(i)
            task_id = upload_file(page, port, page_category, 'uploader', data)
            result[(page.replace(UPLOAD_FOLDER, ""))] = {'class': page_category, 'task_id': task_id, 'url_to_retrieve_result': base_url.format(port, page_category, 'elab_result/'+task_id)}
        else:
            result[(page.replace(UPLOAD_FOLDER, ""))] = {'class': 'other', 'task_id': None, 'url_to_retrieve_result': None}
        i += 1

    if filePath[-4:] == ".pdf":
        pattern = (filePath.replace(".pdf","")) + "*"
    elif filePath[-4:] == ".PDF":
        pattern = (filePath.replace(".PDF","")) + "*"
    # cleanup(pattern)

    return result