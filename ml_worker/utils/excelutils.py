import os
import re
from datetime import datetime as dt
from dateutil.parser import parse
from thefuzz import fuzz

UPLOAD_FOLDER = '/flask_app/files/xlsx/'
import pandas as pd


def format_doc(doc_type, doc_name, extracted_data, pathfile):
    if doc_type == 'vgm':
        return format_vgm(doc_name, extracted_data)
    elif doc_type == 'bolinstruction':
        return format_bolinstruction(doc_name, extracted_data)
    else:
        return


def prune_text(text):
    chars = "\\`*_\{\}[]\(\)\|/<>#-\'\"+!$,\."
    for c in chars:
        if c in text:
            text = text.replace(c, "")
    return text


def cleanup_text(text):
    result = re.sub(r'[^a-zA-Z0-9]+', '', text)
    return result


def format_vgm(doc_name, extracted_data):
    """
    EXAMPLE extracted_data
    {
       "0":{
          "detection_index":"0.80",
          "data_to_review":[
             {
                "key":"Header",
                "type":"Inputs",
                "value":[
                   {
                      "key":"booking number",
                      "value":"CNAN12961",
                      "state":"INCOMPLETE"
                   },
                   {
                      "key":"container id",
                      "value":"EMAU3021997",
                      "state":"INCOMPLETE"
                   },
                   {
                      "key":"signer name",
                      "value":"DANIELA  FASANELLA",
                      "state":"INCOMPLETE"
                   },
                   {
                      "key":"shipper",
                      "value":"COMPANY SPA",
                      "state":"INCOMPLETE"
                   },
                   {
                      "key":"vgm",
                      "value":"Kg  27260",
                      "state":"INCOMPLETE"
                   }
                ],
                "page":1
             }
          ]
       },
       "1":{
          "detection_index":"0.80",
          "data_to_review":[
             {
                "key":"Header",
                "type":"Inputs",
                "value":[
                   {
                      "key":"container id",
                      "value":"EMAU3021997",
                      "state":"INCOMPLETE"
                   },
                   {
                      "key":"vgm",
                      "value":"27.260",
                      "state":"INCOMPLETE"
                   }
                ],
                "page":2
             }
          ]
       }
    }
    """

    doc_name_contents = re.split("_", doc_name, 2)
    if len(doc_name_contents) == 3:
        attach_filename = doc_name_contents[2]
    else:
        attach_filename = doc_name
    reference_number = attach_filename.replace(".PDF", "").replace(".pdf", "")

    containerId, bookingNumber, authorizedPerson, vgm, signerName, shipper, containerType = [], [], [], [], [], [], []
    for page_nr in extracted_data:
        print(page_nr)
        data_to_review = extracted_data[page_nr]['data_to_review']
        for element in data_to_review:
            if element['key'] == 'Header':
                page = element['page']
                print('page: {}'.format(page))
                for element_item in element['value']:
                    # print(element_item['key'])
                    if element_item['key'] == 'container id' and element_item['value'] != "":
                        value = cleanup_text(element_item['value'])
                        containerId.append(value)
                    if element_item['key'] == 'booking number' and element_item['value'] != "":
                        value = cleanup_text(element_item['value'])
                        bookingNumber.append(value)
                    if element_item['key'] == 'authorized person' and element_item['value'] != "":
                        authorizedPerson.append(element_item['value'])
                    if element_item['key'] == 'vgm' and element_item['value'] != "":
                        vgm.append(element_item['value'])
                    if element_item['key'] == 'signer name' and element_item['value'] != "":
                        signerName.append(element_item['value'])
                    if element_item['key'] == 'shipper' and element_item['value'] != "":
                        shipper.append(element_item['value'])
                    if element_item['key'] == 'container type' and element_item['value'] != "":
                        containerType.append(element_item['value'])

    xls_filepath = os.path.join(UPLOAD_FOLDER, reference_number + ".xlsx")
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # CURRENT EXCEL HEADER
    # num. Booking | Tipo Cnt (Codice Iso) | Num. Container | Peso VGM | Nome persona autorizzata al Vgm | Caricatore/Shipper | Metodo 1 (Conservo scontrino) | Metodo 1 (allego scontrino) | Metodo 2(certificazione AEO o ISO9001/28000)
    df = pd.DataFrame({
                       'num. Booking': pd.Series(bookingNumber),
                       'Tipo Cnt': pd.Series(containerType),
                       'Num. Container': pd.Series(containerId),
                       'Peso VGM': pd.Series(vgm),
                       'Nome persona autorizzata al Vgm': pd.Series(authorizedPerson),
                       'Nome persona che ha firmato il Vgm': pd.Series(signerName),
                       'Shipper': pd.Series(shipper)
                       })

    writer = pd.ExcelWriter(xls_filepath, engine='xlsxwriter')
    df.to_excel(writer, sheet_name="Sheet1", index=False)  # send df to writer
    worksheet = writer.sheets["Sheet1"]  # pull worksheet object
    for idx, col in enumerate(df):  # loop through all columns
        series = df[col]
        max_len = max((
            series.astype(str).map(len).max(),  # len of largest item
            len(str(series.name))  # len of column name/header
            )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len)  # set column width
    writer.close()

    # df.to_excel(xls_filepath, index=False)

    print('excel file has been created!')
    print(df)

    return xls_filepath, reference_number + ".xlsx"


def format_bolinstruction(doc_name, extracted_data):
    """
    EXAMPLE extracted_data
    {
  "data_to_review": [
    [
      {
        "key": "Header",
        "page": 1,
        "type": "Inputs",
        "value": [
          {
            "key": "shipper",
            "state": "INCOMPLETE",
            "value": "LANDI  RENZO  S.P.A.  VIA  NOBEL,  2  42025  CORTE  TEGGE  -  CAVRIAGO  (RE)  ITALY"
          },
          {
            "key": "consignee",
            "state": "INCOMPLETE",
            "value": "SARL  GHAZAL  ZONE  INDUSTRIELLE  LOT  N°  57  B  16000  OUED  -  SMAR  /  ALGIERS  ALGERIE  PHONE:  0021321233667  NIF:  099916000622665"
          },
          {
            "key": "notify",
            "state": "INCOMPLETE",
            "value": "SARL  GHAZAL  ZONE  INDUSTRIELLE  LOT  N°  57  B  16000  OUED  -  SMAR  /  ALGIERS  ALGERIE  PHONE:  0021321233667  NIF:  099916000622665"
          },
          {
            "key": "portofloading",
            "state": "INCOMPLETE",
            "value": "DJANET  LA  SPEZIA  PORT"
          },
          {
            "key": "portofdischarge",
            "state": "INCOMPLETE",
            "value": "PORT  D��  ALGER"
          },
          {
            "key": "containerquantity",
            "state": "INCOMPLETE",
            "value": ""
          },
          {
            "key": "containertype",
            "state": "INCOMPLETE",
            "value": "1X40��  BOX  1X20��"
          },
          {
            "key": "containerid",
            "state": "INCOMPLETE",
            "value": ""
          },
          {
            "key": "sealnumber",
            "state": "INCOMPLETE",
            "value": "SGCU8692557  SEAL  1600762  BOX  EMAU3043954  SEAL  1600915"
          },
          {
            "key": "packagequantity",
            "state": "INCOMPLETE",
            "value": "21  PALLETS  10  PALLETS  H.S.CODE  8409  9100"
          },
          {
            "key": "grossweight",
            "state": "INCOMPLETE",
            "value": "KGS  KGS"
          },
          {
            "key": "hscode",
            "state": "INCOMPLETE",
            "value": ""
          },
          {
            "key": "contentdescription",
            "state": "INCOMPLETE",
            "value": "GAS  SYSTEMS  KIT  FOR  CARS  30  DAYS  FREE  TIME  AT  DESTINATION  FREIGHT  PREPAID"
          },
          {
            "key": "incoterm",
            "state": "INCOMPLETE",
            "value": ""
          },
          {
            "key": "lc_number",
            "state": "INCOMPLETE",
            "value": ""
          },
          {
            "key": "date_of_issue",
            "state": "INCOMPLETE",
            "value": ""
          },
          {
            "key": "latestdateofshipment",
            "state": "INCOMPLETE",
            "value": ""
          },
          {
            "key": "cad",
            "state": "INCOMPLETE",
            "value": ""
          },
          {
            "key": "incoterms",
            "state": "INCOMPLETE",
            "value": ""
          }
        ]
      }
    ]
  ],
  "detection_index": 0.8
}
"""

    doc_name_contents = re.split("_", doc_name, 2)
    if len(doc_name_contents) == 3:
        attach_filename = doc_name_contents[2]
    else:
        attach_filename = doc_name
    reference_number = attach_filename.replace(".PDF", "").replace(".pdf", "")

    # 'shipper', 'consignee', 'notify', 'portofloading', 'portofdischarge',
    # 'containertype', 'containerid', 'sealnumber', 'packagequantity',
    # 'contentdescription', 'grossweight', 'hscode', 'deliveryterms',
    # 'lcnumber', 'dateofissue', 'cad', 'dimension'
    shipper, consignee, notify, portofloading, portofdischarge = [], [], [], [], []
    containertype, containerid, sealnumber, packagequantity = [], [], [], []
    contentdescription, grossweight, hscode, deliveryterms = [], [], [], []
    incoterm, lc_number, date_of_issue, latestdateofshipment,cad,containerquantity = [], [], [], [],[],[]
    for page_nr in extracted_data:
        print(page_nr)
        data_to_review = extracted_data[page_nr]['data_to_review']
        for element in data_to_review:
            if element['key'] == 'Header':
                page = element['page']
                print('page: {}'.format(page))
                for element_item in element['value']:
                    # print(element_item['key'])
                    if element_item['key'] == 'shipper' and element_item['value'] != "":
                        shipper.append(element_item['value'])
                    if element_item['key'] == 'consignee' and element_item['value'] != "":
                        consignee.append(element_item['value'])
                    if element_item['key'] == 'notify' and element_item['value'] != "":
                        notify.append(element_item['value'])
                    if element_item['key'] == 'portofloading' and element_item['value'] != "":
                        portofloading.append(element_item['value'])
                    if element_item['key'] == 'portofdischarge' and element_item['value'] != "":
                        portofdischarge.append(element_item['value'])
                    if element_item['key'] == 'containerquantity' and element_item['value'] != "":
                        # value = cleanup_text(element_item['value'])
                        containerquantity.append(value)
                    if element_item['key'] == 'containertype' and element_item['value'] != "":
                        value = cleanup_text(element_item['value'])
                        containertype.append(value)
                    if element_item['key'] == 'containerid' and element_item['value'] != "":
                        containerid.append(element_item['value'])
                    if element_item['key'] == 'sealnumber' and element_item['value'] != "":
                        sealnumber.append(element_item['value'])
                    if element_item['key'] == 'packagequantity' and element_item['value'] != "":
                        packagequantity.append(element_item['value'])
                    if element_item['key'] == 'grossweight' and element_item['value'] != "":
                        grossweight.append(element_item['value'])
                    if element_item['key'] == 'hscode' and element_item['value'] != "":
                        hscode.append(element_item['value'])
                    if element_item['key'] == 'contentdescription' and element_item['value'] != "":
                        contentdescription.append(element_item['value'])
                    if element_item['key'] == 'incoterm' and element_item['value'] != "":
                        incoterm.append(element_item['value'])
                    if element_item['key'] == 'lc_number' and element_item['value'] != "":
                        lc_number.append(element_item['value'])
                    if element_item['key'] == 'date_of_issue' and element_item['value'] != "":
                        date_of_issue.append(element_item['value'])
                    if element_item['key'] == 'latestdateofshipment' and element_item['value'] != "":
                        latestdateofshipment.append(element_item['value'])
                    if element_item['key'] == 'cad' and element_item['value'] != "":
                        cad.append(element_item['value'])

    xls_filepath = os.path.join(UPLOAD_FOLDER, reference_number + ".xlsx")
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # CURRENT EXCEL HEADER
    # MERCE | TIPOLOGIA CONTAINER | SIGLA | COLLI | PESO LORDO | PESO NETTO | VOLUME | SIGILLI | TIPO | IMBALLO | TARA
    df = pd.DataFrame({
                       'shipper': pd.Series(shipper),
                       'consignee': pd.Series(consignee),
                       'notify': pd.Series(notify),
                       'portofloading': pd.Series(portofloading),
                       'portofdischarge': pd.Series(portofdischarge),
                       'containertype': pd.Series(containertype),
                       'containerid': pd.Series(containerid),
                       'sealnumber': pd.Series(sealnumber),
                       'packagequantity': pd.Series(packagequantity),
                       'grossweight': pd.Series(grossweight),
                       'hscode': pd.Series(hscode),
                       'contentdescription': pd.Series(contentdescription),
                       'incoterm': pd.Series(incoterm),
                       'lc_number': pd.Series(lc_number),
                       'date_of_issue': pd.Series(date_of_issue),
                       'latestdateofshipment': pd.Series(latestdateofshipment),
                       'cad': pd.Series(cad)
                       })

    writer = pd.ExcelWriter(xls_filepath, engine='xlsxwriter')
    df.to_excel(writer, sheet_name="Sheet1", index=False)  # send df to writer
    worksheet = writer.sheets["Sheet1"]  # pull worksheet object
    for idx, col in enumerate(df):  # loop through all columns
        series = df[col]
        max_len = max((
            series.astype(str).map(len).max(),  # len of largest item
            len(str(series.name))  # len of column name/header
            )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len)  # set column width
    writer.close()

    # df.to_excel(xls_filepath, index=False)

    print('excel file has been created!')
    print(df)

    return xls_filepath, reference_number + ".xlsx"