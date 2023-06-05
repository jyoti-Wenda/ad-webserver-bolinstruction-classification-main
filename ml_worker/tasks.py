import time
import os
import re
from stat import *
from celery import Celery
from celery.utils.log import get_task_logger
from utils import AWSutils
from utils import excelutils as xlsutils
from utils import webhookutils as whutils
from utils import layoutLMutils as lmutils
from utils import SMTPutils as smtputils

UPLOAD_FOLDER = '/flask_app/files/'
s3BucketName = "activedocumentsbucket"
s3BucketRegion = "us-east-1"
serviceDirectory = 'activedocuments/bolinstruction/pdf'
doc_type = 'TEST'
# doc_type = 'BOL'

logger = get_task_logger(__name__)

app = Celery('tasks',
             broker='amqp://admin:mypass@rabbit:5672',
             backend='rpc://')

class perioli_email_set:
    sender = {
        'VGM': [['Wenda Active Documents'], ['vgm@wenda-it.com'], ['zladwbhhjfbnldic'], ['smtp.gmail.com']],
        'BOL': [['Wenda Active Documents'], ['vgm@wenda-it.com'], ['zladwbhhjfbnldic'], ['smtp.gmail.com']],
        'MAN': [['Wenda Active Documents'], ['vgm@wenda-it.com'], ['zladwbhhjfbnldic'], ['smtp.gmail.com']],
        'TEST': [['Wenda Active Documents'], ['vgm@wenda-it.com'], ['zladwbhhjfbnldic'], ['smtp.gmail.com']]
        }

    receiver = {
        'VGM': [['Federico Natale','Giulia Rossi'], ['f.natale@cnanitalia.it','giuliarossi@darioperioli.it']],
        'BOL': [['Noemi Valerio','L. Pardini'], ['n.valerio@cnanitalia.it','l.pardini@cnanitalia.it']],
        'MAN': [],
        'TEST': [['Valentina Protti'], ['valentina@wenda-it.com']]
        }

# app.conf.broker_pool_limit = 0
app.conf.task_publish_retry = False
app.conf.broker_heartbeat = 1800

# @app.task()
# def elab_file(user, filePath, save, output, ocr,webhook, pathfile, localpath, send_email=True):
#     logger.info('ASYNC POST /uploader > Got Request - Starting work')
#     print("filepath is: {}".format(filePath))
#     if os.path.isfile(filePath):
#         print('I can read this file path!')
#         if save:
#             AWSutils.uploadToBucket(s3BucketName, s3BucketRegion, serviceDirectory, UPLOAD_FOLDER, user, os.path.basename(filePath))
#         formatted_result, result = lmutils.elab(filePath,ocr)
#         if output.lower() == 'excel':
#             output_file, output_name = xlsutils.format_doc('bolinstruction', os.path.basename(filePath), result, pathfile)
#             doc_name_contents = re.split("_", os.path.basename(filePath), 2)
#             if len(doc_name_contents) == 3:
#                 orig_sender = doc_name_contents[0]
#                 try:
#                     orig_mail_sender_regex = re.search(r"([a-zA-Z0-9\._]+)\-([a-zA-Z0-9\._\-]+)\-", orig_sender)
#                     orig_mail_sender = orig_mail_sender_regex.group(1) + " (" + orig_mail_sender_regex.group(2) + ")"
#                 except:
#                     orig_mail_sender = orig_sender
#                 attach_pdfname = doc_name_contents[2]
#             else:
#                 attach_pdfname = os.path.basename(filePath)
#             logger.info(output_name)
#             logger.info(output_file)
#             if webhook != '' and pathfile != '':
#                 r = whutils.send_result(webhook, pathfile, localpath, output_file, output_name, filePath, attach_pdfname)
#                 logger.info('webhook {}, result {}'.format(webhook, r))
#                 logger.info('Work Finished')
#                 #return output_file, output_name, 'xls'
#             elif send_email:
#                 smtputils.send_result(perioli_email_set.sender[doc_type][0], perioli_email_set.sender[doc_type][1], perioli_email_set.sender[doc_type][2], perioli_email_set.sender[doc_type][3],
#                                       perioli_email_set.receiver[doc_type][0], perioli_email_set.receiver[doc_type][1],
#                                       doc_type, output_file, output_name, orig_mail_sender,
#                                       filePath, attach_pdfname)
#                 return output_file, output_name, 'xls'
#             else:
#                 logger.info('Work Finished')
#                 return output_file, output_name, 'xls'
#         else:
#             logger.info('Work Finished')
#             return formatted_result, result, 'json'
#     logger.info('Work Finished')
#     return filePath



@app.task()
def elab_file(user, filePath, save, output, ocr, webhook, pathfile, localpath):
    logger.info('ASYNC POST /uploader > Got Request - Starting work')
    print(filePath)
    if os.path.isfile(filePath):
        print('I can read this file path!')
        if save:
            AWSutils.uploadToBucket(s3BucketName, s3BucketRegion, serviceDirectory, UPLOAD_FOLDER, user, os.path.basename(filePath))
        formatted_result, result = lmutils.elab(filePath, ocr)
        if output.lower() == 'mylogsupply':
            logger.info('Sending the classified doc to their elab endpoints')
            task_ids = lmutils.send_to_elaboration(filePath, result, webhook, pathfile, localpath)
            logger.info(task_ids)
            # if needed we can also collect the results from the model extraction and send them back here
            return task_ids, result, 'json'
        else:
            logger.info('Work Finished')
            return formatted_result, result, 'json'
    logger.info('Work Finished')
    return filePath, os.path.basename(filePath), 'pdf'