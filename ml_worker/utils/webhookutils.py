import requests

def send_result(webhook, pathfile, localpath, output_file, output_name, input_file, input_name):
    if input_file != "":
        files = {'file': (output_name, open(output_file, 'rb')),
                 'pdf': (input_name, open(input_file, 'rb')),}
    else:
        files = {'file': (output_name, open(output_file, 'rb'))}
    values = {'pathfile': pathfile,
              'localpath': localpath}
    r = requests.post(webhook, files=files, data=values)
    return r