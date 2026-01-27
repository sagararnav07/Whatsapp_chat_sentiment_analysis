import re
import pandas as pd

def preprocess(data):
    # Try different WhatsApp export formats
    # Format 1: [dd/mm/yy, h:mm:ss AM/PM] Username: message (iOS/newer Android)
    # Format 2: dd/mm/yy, h:mm AM/PM - Username: message (older Android)
    
    # Pattern for bracket format: [10/10/25, 7:32:49 AM] Username: message
    bracket_pattern = r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s?[APap][Mm])\]\s'
    
    # Pattern for dash format: 10/10/25, 7:32 AM - Username: message
    dash_pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s?[APap]?[Mm]?)\s-\s'
    
    # Detect which format is used
    if re.search(bracket_pattern, data):
        pattern = bracket_pattern
        date_formats = [
            '%d/%m/%y, %I:%M:%S %p',
            '%d/%m/%Y, %I:%M:%S %p',
            '%m/%d/%y, %I:%M:%S %p',
            '%m/%d/%Y, %I:%M:%S %p',
            '%d/%m/%y, %I:%M:%S %p',
            '%d/%m/%Y, %H:%M:%S',
            '%d/%m/%y, %H:%M:%S',
        ]
    else:
        pattern = dash_pattern
        date_formats = [
            '%d/%m/%Y, %H:%M',
            '%d/%m/%y, %H:%M',
            '%m/%d/%Y, %H:%M',
            '%m/%d/%y, %H:%M',
            '%d/%m/%Y, %I:%M %p',
            '%d/%m/%y, %I:%M %p',
            '%m/%d/%Y, %I:%M %p',
            '%m/%d/%y, %I:%M %p',
        ]

    messages = re.split(pattern, data)[1:]
    
    # Extract dates and messages alternately
    dates = messages[0::2]
    user_messages = messages[1::2]

    df = pd.DataFrame({'user_message': user_messages, 'message_date': dates})
    
    # Try multiple date formats to handle different WhatsApp export formats
    for fmt in date_formats:
        try:
            df['message_date'] = pd.to_datetime(df['message_date'], format=fmt)
            break
        except (ValueError, TypeError):
            continue
    else:
        # Fallback: let pandas infer the format
        df['message_date'] = pd.to_datetime(df['message_date'], format='mixed', dayfirst=True)

    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        # Handle both "Username: message" format
        entry = re.split(r'^([^:]+):\s', message, maxsplit=1)
        if len(entry) >= 3:  # Successfully split: ['', 'username', 'message']
            users.append(entry[1].strip())
            messages.append(entry[2].strip())
        else:
            users.append('group_notification')
            messages.append(message.strip())

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df