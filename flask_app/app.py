from flask import Flask, render_template, request, redirect, make_response, send_file, Blueprint
from celery import Celery
from werkzeug.utils import secure_filename
import os
import stat

app = Flask(__name__)
async_app = Celery('ml_worker',
                    broker='amqp://admin:mypass@rabbit:5672',
                    backend='rpc://')

UPLOAD_FOLDER = '/flask_app/files/'
bp = Blueprint('bolinstruction', __name__,
               template_folder='templates')


@bp.route('/')
def base():
    return {'error': 'no endpoint specified'}, 400


@bp.route('/echo')
def echo():
    return "Hello! FINALMENTE!", 200


@bp.route('/upload')
def upload():
    headers = {'Content-Type': 'text/html'}
    return make_response(render_template('upload-bolinstruction.html'), 200, headers)


@bp.route('/uploader', methods = ['POST'])
def async_uploader():
    print("Invoking Method to start elaboration")
    u = request.values.get('user', '')
    f = request.files.get('file', None)
    s = request.form.get('save', False)
    output = request.form.get('output', None)
    ocr = request.form.get('ocr', '')
    webhook = request.form.get('webhook', '')  # ti mando url dove mandarmi il file sempre nel formato formdata
    pathfile = request.form.get('pathfile', '')  # dove c'è percorso+filename del file che ti ho inviato e questo me lo devi rimandare insieme al file
    localpath = request.form.get('localpath', '')  # anche questo è da restituire
    if f != None:
        f.save(os.path.join(UPLOAD_FOLDER, secure_filename(f.filename)))
        os.chmod(os.path.join(UPLOAD_FOLDER, secure_filename(f.filename)), stat.S_IRWXO)
        if any(ext in f.filename.lower() for ext in [".pdf", ".jpg", ".png", ".jpeg", ".docx", ".doc","msg","xlsx","xls",".XLS"]):
            r = async_app.send_task('tasks.elab_file', kwargs={'user': u,
                                                               'filePath': os.path.join(UPLOAD_FOLDER, secure_filename(f.filename)),
                                                               'save': s,
                                                               'output': output,
                                                               'ocr': ocr,
                                                               'webhook': webhook,
                                                               'pathfile': pathfile,
                                                               'localpath': localpath})
            app.logger.info(r.backend)
            return {'task_id': r.id}, 200
        else:
            return {'error': 'invalid file extension (accepted files: pdf, jpg, png, docx)'}, 401
    else:
        return {'error': 'file not found'}, 401


@bp.route('/elab_status/<task_id>')
def elab_status(task_id):
    status = async_app.AsyncResult(task_id, app=async_app)
    print("Invoking Method to get task status")
    print(task_id)
    print(status)
    return "Status of the Task " + str(status.state)


@bp.route('/elab_result/<task_id>')
def elab_result(task_id):
    output_file, output_name, format = async_app.AsyncResult(task_id).result
    # return "Result of the Task " + str(result)
    if format == 'xls':
        if os.path.isfile(output_file):
            return send_file(output_file,
                            download_name=output_name,
                            as_attachment=True)
        else:
            return {'error': 'file not found'}, 401
    elif format == 'json':
        return make_response(output_file, 200)


app.register_blueprint(bp, url_prefix='/activedocuments/bolinstruction')