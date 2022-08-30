import smtplib, ssl
from email.mime.text import MIMEText
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
    
    def send_email(self, subject, start_time, password="jmfzjrnyooyrhteo", 
                   port=465, sender_email=None, 
                   receiver_email='monte.flora@noaa.gov'):
        """
        Send an email when a job is finished is running. 
        
        Parameters
        -----------------
        subject : str 
            The subject of the email. 
        
        start_time : datetime.datetime object 
            The date and time when the job was started.
            
        password : str 
            The password of the sender's email. 
        
        port : int
        
        sender_email : str (default=None)
            The email address that will be sending the message.
            By default the sender email is set as the receiver's email.
            
        receiver_email : str 
            The email address that will be sending the message.
        
        
        """
        if sender_email is None:
            sender_email = receiver_email
        
        hours, minutes, seconds = self.get_duration(start_time)
        
        message = f""" 
               The job has finished! 
               {subject}
               Started at {start_time.strftime("%Y-%m-%d  %I:%M %p")}, 
               Duration : {hours:.2f} hours : {minutes:.2f} minutes : {seconds:.2f} seconds
              """

        # Create a secure SSL context
        context = ssl.create_default_context()
        
        # Create the subject and message 
        msg = MIMEText(message)
        for key, item in zip(['Subject', 'From', 'To'], [subject, sender_email, receiver_email]):
            msg[key] = item 

        # Send the mesage. 
        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())