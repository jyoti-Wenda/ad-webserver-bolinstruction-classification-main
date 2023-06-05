import boto3
from botocore.errorfactory import ClientError
import time
import pandas as pd
import os
from trp import Document
from dotenv import load_dotenv
import PyPDF2
from datetime import datetime
from xml.etree import ElementTree as ET

load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')


def uploadAndProcess(s3BucketName, s3BucketRegion, serviceDirectory, folder, client, filename, saveFlag):
    uploadToBucket(s3BucketName, s3BucketRegion, serviceDirectory, folder, client, filename)  # duplicate uploads not allowed: filename must be unique!

    resultFolder = os.path.join(folder, filename.lower().replace(".pdf", ""))
    if os.path.isdir(resultFolder):
        print("Document already processed, results in: " + resultFolder)
    else:
        jobId = startJob(s3BucketName, s3BucketRegion, serviceDirectory + '/' + client + '/' + filename)
        print("Started job with id: {}".format(jobId))
        if(isJobComplete(jobId, s3BucketRegion)):
            response = getJobResults(jobId, s3BucketRegion)

        # Print detected text (to files)
        process_text_result(response, folder, filename)

    if not saveFlag:
        cleanupFromBucket(s3BucketName, s3BucketRegion, serviceDirectory, folder, client, filename)
        

# Start a document analysis asynchronous job on an S3 object
def startJob(s3BucketName, s3BucketRegion, objectName):
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    response = None
    client = session.client('textract', region_name=s3BucketRegion)
    response = client.start_document_analysis(
    DocumentLocation={
        'S3Object': {
            'Bucket': s3BucketName,
            'Name': objectName
        }
    },
    FeatureTypes=[
        'TABLES','FORMS',
    ])

    return response["JobId"]


def startJobMOCKUP(s3BucketName, objectName):
    print(s3BucketName)
    print(objectName)
    
    return "JOBID-MOCKUP"


# Check if a document analysis job is completed (loops every 5s and returns only when status is not IN PROGRESS)
def isJobComplete(jobId, s3BucketRegion):
    time.sleep(5)
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    client = session.client('textract', region_name=s3BucketRegion)
    response = client.get_document_analysis(JobId=jobId)
    status = response["JobStatus"]
    print("Job status: {}".format(status))

    while(status == "IN_PROGRESS"):
        time.sleep(5)
        response = client.get_document_analysis(JobId=jobId)
        status = response["JobStatus"]
        print("Job status: {}".format(status))

    return status


def isJobCompleteMOCKUP(jobId):
    time.sleep(5)
    print(jobId)
    status = "COMPLETE"
    print("Job status: {}".format(status))

    return status


# Retrieve the document analysis result (multiple pages accepted)
def getJobResults(jobId, s3BucketRegion):

    pages = []

    time.sleep(5)

    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    client = session.client('textract', region_name=s3BucketRegion)
    response = client.get_document_analysis(JobId=jobId)
    
    pages.append(response)
    print("Resultset page received: {}".format(len(pages)))
    nextToken = None
    if('NextToken' in response):
        nextToken = response['NextToken']

    while(nextToken):
        time.sleep(5)

        response = client.get_document_analysis(JobId=jobId, NextToken=nextToken)

        pages.append(response)
        print("Resultset page received: {}".format(len(pages)))
        nextToken = None
        if('NextToken' in response):
            nextToken = response['NextToken']

    return pages


def getJobResultsMOCKUP(jobId):
    print(jobId)

    return []


# Displays information about a block returned by text detection and text analysis
def DisplayBlockInformation(block):
    print('Id: {}'.format(block['Id']))
    if 'Text' in block:
        print('    Detected: ' + block['Text'])
    print('    Type: ' + block['BlockType'])
   
    if 'Confidence' in block:
        print('    Confidence: ' + "{:.2f}".format(block['Confidence']) + "%")

    if block['BlockType'] == 'CELL':
        print("    Cell information")
        print("        Column:" + str(block['ColumnIndex']))
        print("        Row:" + str(block['RowIndex']))
        print("        Column Span:" + str(block['ColumnSpan']))
        print("        RowSpan:" + str(block['ColumnSpan']))    
    
    if 'Relationships' in block:
        print('    Relationships: {}'.format(block['Relationships']))
    print('    Geometry: ')
    print('        Bounding Box: {}'.format(block['Geometry']['BoundingBox']))
    print('        Polygon: {}'.format(block['Geometry']['Polygon']))
    
    if block['BlockType'] == "KEY_VALUE_SET":
        print ('    Entity Type: ' + block['EntityTypes'][0])
    if 'Page' in block:
        print('Page: ' + str(block['Page']))
    print()


# Prints to csv files the detected key-value pairs and tables
def process_text_result(response, documentFolder, documentName):
    resultFolder = os.path.join(documentFolder, documentName.lower().replace(".pdf", ""))
    if not (os.path.exists(resultFolder)):  # This should be a double-check - it won't get here if the folder exists
        print("Creating folder: " + resultFolder)
        os.mkdir(resultFolder)
    i=0
    for resultPage in response:
        # doc = Document(resultPage)

        #Get the text blocks
        blocks=resultPage['Blocks']
        print ('Detected Document Text')

        df = pd.DataFrame(blocks)
        df.to_csv("{}/{}-{}.csv".format(resultFolder, documentName, i))
        
        i+=1
        # for block in blocks:
        #     DisplayBlockInformation(block)

    doc = Document(response)

    # Write raw data
    rawText = ""
    for page in doc.pages:
        rawText += page.text + "\n"
    with open("{}/{}-rawdata.txt".format(resultFolder, documentName), "w") as f:
        f.write(rawText)

    keys = []
    values = []
    for page in doc.pages:
        # Print fields
        print("Fields:")
        for field in page.form.fields:
            print("Key: {}, Value: {}".format(field.key, field.value))
            try:
                keys.append(field.key.text)
            except:
                keys.append('')
            try:
                values.append(field.value.text)
            except:
                values.append('')
                
    keyvalues = list(zip(keys,values))
    d = pd.DataFrame(keyvalues, columns=['key','value'])
    d.to_csv("{}/{}-keyvalues.csv".format(resultFolder, documentName))

    # Print tables
    j = 0
    for page in doc.pages:
        dataframes = {}
        for table in page.tables:
            tMatrix = []
            for r, row in enumerate(table.rows):
                rList = []
                for c, cell in enumerate(row.cells):
                    # print("Table[{}][{}] = {}".format(r, c, cell.text))
                    rList.append(cell.text)
                tMatrix.append(rList)
            j += 1
            # print("TABLE {}".format(j))
            # print(tMatrix)
            dataframes[j] = pd.DataFrame(tMatrix)
                
        for dfindex in dataframes:
            print(dfindex)
            print(dataframes[dfindex])
            dataframes[dfindex].to_csv("{}/{}-table-{}.csv".format(resultFolder, documentName, dfindex))


# Uploads file to S3 bucket (fileName is both the real file name and the desired name for the S3 object)
def uploadToBucket(bucket, region, serviceDirectory, folder, client, fileName):
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    s3 = session.resource('s3', region_name=region)
    print(bucket)
    print(folder)
    print(fileName)    
    
    try:
        cli = session.client('s3', region_name=region)
        cli.head_object(Bucket=bucket, Key=serviceDirectory + '/' + client + '/' + fileName)
        print("File already uploaded: " + fileName)
    except ClientError:
        try:
            response = s3.Bucket(bucket).upload_file(os.path.join(folder, fileName), serviceDirectory + '/' + client + '/' + fileName)
            print("File uploaded: " + serviceDirectory + '/' + client + '/' + fileName)
        except:
            print("Something went wrong while uploading")


def cleanupFromBucket(bucket, region, serviceDirectory, folder, client, fileName):
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    s3 = session.resource('s3', region_name=region)
    print(bucket)
    print(folder)
    print(fileName)    
    
    try:
        cli = session.client('s3', region_name=region)
        cli.head_object(Bucket=bucket, Key=serviceDirectory + '/' + client + '/' + fileName)
        print("File found: " + fileName)
        s3.Object(bucket, serviceDirectory + '/' + client + '/' + fileName).delete()
        print("File deleted from bucket")
    except ClientError:
        try:
            print("File not found: " + serviceDirectory + '/' + client + '/' + fileName)
        except:
            print("Something went wrong while deleting")


def uploadToBucketMOCKUP(bucket, localPath, fileName):
    print(bucket)
    print(localPath)
    print(fileName)
    
    print("File uploaded: " + fileName)


if __name__ == "__main__":
    # Folder containing the documents
    documentFolder = "/Users/admin/Documents/AWS"
    # Documents
    documentNames = [
        "596-ddt + coa ATOTECH ESP.PDF",
        "594-TPO-48998-0 RHENUS LOGISTICS S.p.A..pdf",
        "589-PACKLIST 148.pdf",
        "588-104069.pdf",
        "582-2022_03_03_001203.Pdf",
        "3563-DDT 641 VESTA.pdf"
    ]
    s3BucketName = "activedocumentsbucket"
    # documentName = "23660-CUTRONE__200797_24022022_.PDF"

    for documentName in documentNames:
        jobId = startJob(s3BucketName, documentName)
        print("Started job with id: {}".format(jobId))
        if(isJobComplete(jobId)):
            response = getJobResults(jobId)

        # Print detected text (to files)
        process_text_result(response, documentFolder, documentName)
        