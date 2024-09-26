from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os, pickle
import base64
import email
import json


# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_mime_message(service, user_id, msg_id):
    """Get a MIME Message using the Gmail API's get method."""
    try:
        message = service.users().messages().get(userId=user_id, id=msg_id, format='raw').execute()
        msg_str = base64.urlsafe_b64decode(message['raw'].encode('ASCII'))
        mime_msg = email.message_from_bytes(msg_str)
        return mime_msg
    except Exception as error:
        print(f'An error occurred: {error}')
        return None

def get_gmail_service():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                '/home/x/Downloads/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)
    return service

def main():

    list_of_emails =[]
 
    service = get_gmail_service()
    # Call the Gmail API to fetch INBOX
    end_date = "2024/03/17"
    start_date = "2024/01/01"
    query = f'from:John Authers <noreply@mail.bloombergbusiness.com> after:{start_date} before:{end_date}'
    response = service.users().messages().list(userId='me', q=query).execute()
    #messages = results.get('messages', [])[:3]
    
    messages =[]
    while 'messages' in response:
        messages.extend(response['messages'])
        if 'nextPageToken' in response:
            page_token = response['nextPageToken']
            response = service.users().messages().list(userId='me', q=query, pageToken=page_token).execute()
        else:
            break


    if not messages:
        print("No messages found.")
    else:
        for message in messages:
            email_data = dict()
            msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()

            headers = msg['payload']['headers']
            date = next(header['value'] for header in headers if header['name'] == 'Date')
            print(f"Date: {date}")
            email_data["date"]= date

            mime_msg = get_mime_message(service, 'me', message['id'])

            if mime_msg.is_multipart():
                for part in mime_msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))

                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        content= part.get_payload(decode=True).decode()  # prints plain text
                        # Remove empty lines
                        content_lines = content.splitlines()
                        content_without_empty_lines = "\n".join(line for line in content_lines if line.strip())

                        # Remove everything from "Survival Tips" onward
                        cutoff_index = content_without_empty_lines.find("Survival Tips")
                        if cutoff_index != -1:  # If "Survival Tips" is found
                            content_final = content_without_empty_lines[:cutoff_index]
                        else:
                            content_final = content_without_empty_lines

                        # content_final now holds the processed content
                        #print(content_final)
                        email_data["content"]= content_final
                        list_of_emails.append(email_data)

                    # elif content_type == "text/html" and "attachment" not in content_disposition:
                    #     print(part.get_payload(decode=True).decode())  # prints HTML text
            else:
                # if the email message is not multipart
                print(mime_msg.get_payload(decode=True).decode())

    # Convert the dictionary to a JSON string
    email_data_json = json.dumps(list_of_emails, indent=4,ensure_ascii=False)

    # Define the filename where the JSON data will be saved
    filename = 'email_data.json'

    # Write the JSON string to a file
    with open(filename, 'w',encoding='utf-8') as file:
        file.write(email_data_json)

    print(f"Data written to {filename}")

if __name__ == '__main__':
    main()
