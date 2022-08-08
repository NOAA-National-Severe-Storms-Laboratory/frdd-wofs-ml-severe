import smtplib, ssl
import datetime

class Emailer:
    def get_start_time(self):
        return datetime.datetime.now()
    
    def get_duration(self, start_time):
        duration =  datetime.datetime.now() - start_time
        seconds = duration.total_seconds()
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
    
        return hours, minutes, seconds
    
    def send_message(self, message, start_time):
        hours, minutes, seconds = self.get_duration(start_time)
        
        message = f""" 
               Subject: Finished!
               {message}
               Started at {start_time.strftime("%I:%M %p")}, 
               Duration : {hours:.2f} hours : {minutes:.2f} minutes : {seconds:.2f} seconds
              """
        self.send_email(message)

    def send_email(self, message):
        port = 465  # For SSL
        password = "jmfzjrnyooyrhteo"

        # Create a secure SSL context
        context = ssl.create_default_context()
        sender_email = "monte.flora@noaa.gov"
        receiver_email = "monte.flora@noaa.gov"

        base_message = 'From : {sender_email}\r\nTo: {receiver_email}\r\n\r\n'

        base_message = base_message + message

        print(f'{base_message=}') 

        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login("monte.flora@noaa.gov", password)
            server.sendmail(sender_email, receiver_email, base_message)
