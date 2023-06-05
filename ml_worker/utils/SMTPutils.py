import smtplib, email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication


def update_logger(sender_name, sender_email, sender_pwd, sender_host, receiver_name, receiver_email, message_content):
    message = """\
From: {0} <{1}>
To: {2} <{3}>
Subject: {1} Active Email - logger updated

{4}
    """

    try:
        smtpObj = smtplib.SMTP(sender_host)
        smtpObj.ehlo()
        smtpObj.starttls()
        smtpObj.ehlo()
        smtpObj.login(sender_email, sender_pwd)
        smtpObj.sendmail(sender_email, receiver_email, message.format(sender_name, sender_email, receiver_name, receiver_email, message_content))
        print("Successfully sent email")
    except smtplib.SMTPException:
        print("Error: unable to send email")
    finally:
        smtpObj.close()


def send_discarded(sender_name, sender_email, sender_pwd, sender_host, receiver_name, receiver_email, message_content, reason):
    with open(message_content) as data:
        # email_data = data[0][1]
        # message = email.message_from_string(email_data)
        message = email.message_from_string(data.read())
        new_subject = "#scarto: " + message['Subject'] + " [" + str(reason) + "]"
        del message["From"]
        del message["To"]
        del message["Subject"]
        del message["CC"]
        message["From"] = sender_name + " <" + sender_email + ">"
        message["To"] = receiver_name + " <" + receiver_email + ">"
        message["Subject"] = new_subject
        message["CCN"] = 'valentina@wenda-it.com'
    try:
        smtpObj = smtplib.SMTP_SSL(sender_host, 465)
        # smtpObj = smtplib.SMTP()
        # smtpObj.ehlo()
        # smtpObj.starttls()
        # smtpObj.ehlo()
        smtpObj.login(sender_email, sender_pwd)
        smtpObj.sendmail(sender_email, receiver_email, message.as_string())
        print("Successfully sent email")
        smtpObj.close()
    except smtplib.SMTPException as e:
        print("Error: unable to send email")
        print(e)
    finally:
        smtpObj.close()


def send_result(sender_name, sender_email, sender_pwd, sender_host,
                receivers_name, receivers_email,
                doc_type, attach_file, attach_filename, orig_mail_sender,
                attach_pdf, attach_pdfname):
    msg = MIMEMultipart()

    to_list = []
    for name, email in zip(receivers_name, receivers_email):
        to_list.append("{0} <{1}>".format(name, email))

    text = """Salve,

In allegato il file in formato excel elaborato in seguito alla ricezione del {}:
{}

Proveniente da:
{}

In caso di dubbi o necessità di chiarimenti contattare:
Luca Boarini
Customer Success Manager
E: luca@wenda-it.com
M: +39 3456416252

Cordiali saluti,
Wenda Active Documents
""".format(doc_type, attach_pdfname, orig_mail_sender)

    msg['From'] = sender_name[0] + " <" + sender_email[0] + ">"
    msg['To'] = ', '.join(to_list)
    msg['Subject'] = "Wenda Active Documents - {} - {} - {}".format(doc_type, attach_pdfname, orig_mail_sender)

    # ccn = "Wenda Notification <notification@wenda-it.com>"
    # receivers_email.append(ccn)

    msg.attach(MIMEText(text))

    with open(attach_file, "rb") as fil:
        ext = attach_file.split('.')[-1:]
        attachedfile = MIMEApplication(fil.read(), _subtype = ext)
        attachedfile.add_header(
            'content-disposition', 'attachment', filename=attach_filename )
    with open(attach_pdf, "rb") as pdf:
        ext = attach_pdf.split('.')[-1:]
        attachedpdf = MIMEApplication(pdf.read(), _subtype = ext)
        attachedpdf.add_header(
            'content-disposition', 'attachment', filename=attach_pdfname )

    msg.attach(attachedfile)
    msg.attach(attachedpdf)

    try:
        smtpObj = smtplib.SMTP(sender_host[0], port= 587)
        smtpObj.starttls()
        smtpObj.login(sender_email[0], sender_pwd[0])
        smtpObj.sendmail(sender_email[0], receivers_email, msg.as_string())
        smtpObj.close()
    except smtplib.SMTPException:
        print("Error: unable to send email")
    finally:
        smtpObj.close()


if __name__ == "__main__":
    # send_discarded("Valentina Protti",
    #                "valentina@wenda-it.com",
    #                "Sww0Z_pS8",
    #                "smtp.gmail.com",
    #                "Protti Valentina",
    #                "protti.valentina@gmail.com",
    #                "/Users/admin/Downloads/Prova.eml")

    send_result("Valentina Protti",
                   "valentina@wenda-it.com",
                   "Sww0Z_pS8",
                   "smtp.gmail.com",
                   ["Protti Valentina"],
                   ["protti.valentina@gmail.com"],
                   "VGM",
                   "/Users/admin/Downloads/Perioli/Vgm6.xlsx",
                   "Test VGM.xlsx")