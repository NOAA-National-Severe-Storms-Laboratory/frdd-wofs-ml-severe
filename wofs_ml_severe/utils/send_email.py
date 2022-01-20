import smtplib, ssl

def send_email(message):
    port = 465  # For SSL
    password = 'bpboiqrqpvzmdqif'

    # Create a secure SSL context
    context = ssl.create_default_context()
    sender_email = "monte.flora@noaa.gov"
    receiver_email = "monte.flora@noaa.gov"

    base_message = 'From : {sender_email}\r\nTo: {receiver_email}\r\n\r\n'

    base_message = base_message + message

    print(base_message) 

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login("monte.flora@noaa.gov", password)
    
        server.sendmail(sender_email, receiver_email, base_message)

