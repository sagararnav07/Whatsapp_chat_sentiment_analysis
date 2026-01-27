from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from textblob import TextBlob
import re

extract = URLExtract()

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages - handle multiple formats
    media_patterns = ['<Media omitted>', 'image omitted', 'video omitted', 'audio omitted', 'sticker omitted', 'document omitted', 'GIF omitted']
    num_media_messages = df[df['message'].str.contains('|'.join(media_patterns), case=False, na=False)].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df_percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index()
    df_percent.columns = ['name', 'percent']
    return x, df_percent

def create_wordcloud(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    f.close()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification'].copy()
    # Filter out media messages
    media_patterns = ['<Media omitted>', 'image omitted', 'video omitted', 'audio omitted', 'sticker omitted', 'document omitted']
    temp = temp[~temp['message'].str.contains('|'.join(media_patterns), case=False, na=False)]

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white',
                   colormap='viridis', max_words=200)
    temp['message'] = temp['message'].apply(remove_stop_words)
    text = temp['message'].str.cat(sep=" ")
    if text.strip():
        df_wc = wc.generate(text)
        return df_wc
    return None

def most_common_words(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    f.close()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification'].copy()
    media_patterns = ['<Media omitted>', 'image omitted', 'video omitted', 'audio omitted', 'sticker omitted', 'document omitted']
    temp = temp[~temp['message'].str.contains('|'.join(media_patterns), case=False, na=False)]

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

# ============== NEW AMAZING FEATURES ==============

def get_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    try:
        analysis = TextBlob(str(text))
        if analysis.sentiment.polarity > 0.1:
            return 'Positive'
        elif analysis.sentiment.polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    except:
        return 'Neutral'

def get_sentiment_score(text):
    """Get sentiment polarity score"""
    try:
        analysis = TextBlob(str(text))
        return analysis.sentiment.polarity
    except:
        return 0

def sentiment_analysis(selected_user, df):
    """Perform sentiment analysis on messages"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['user'] != 'group_notification'].copy()
    media_patterns = ['image omitted', 'video omitted', 'audio omitted', 'sticker omitted', 'document omitted', 'Contact card omitted', 'Missed voice call']
    temp = temp[~temp['message'].str.contains('|'.join(media_patterns), case=False, na=False)]
    
    temp['sentiment'] = temp['message'].apply(get_sentiment)
    temp['sentiment_score'] = temp['message'].apply(get_sentiment_score)
    
    sentiment_counts = temp['sentiment'].value_counts()
    avg_sentiment = temp['sentiment_score'].mean()
    
    return sentiment_counts, avg_sentiment, temp

def user_sentiment_comparison(df):
    """Compare sentiment across users"""
    temp = df[df['user'] != 'group_notification'].copy()
    media_patterns = ['image omitted', 'video omitted', 'audio omitted', 'sticker omitted', 'document omitted']
    temp = temp[~temp['message'].str.contains('|'.join(media_patterns), case=False, na=False)]
    
    temp['sentiment_score'] = temp['message'].apply(get_sentiment_score)
    user_sentiment = temp.groupby('user')['sentiment_score'].mean().sort_values(ascending=False)
    
    return user_sentiment

def emoji_analysis(selected_user, df):
    """Analyze emoji usage"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    
    emoji_df = pd.DataFrame(Counter(emojis).most_common(20), columns=['Emoji', 'Count'])
    return emoji_df

def response_time_analysis(df):
    """Analyze response times between users"""
    df_sorted = df.sort_values('date').copy()
    df_sorted['prev_user'] = df_sorted['user'].shift(1)
    df_sorted['prev_time'] = df_sorted['date'].shift(1)
    df_sorted['response_time'] = (df_sorted['date'] - df_sorted['prev_time']).dt.total_seconds() / 60  # in minutes
    
    # Only consider responses (when user changes) within 24 hours
    responses = df_sorted[(df_sorted['user'] != df_sorted['prev_user']) & 
                          (df_sorted['response_time'] < 1440) &  # 24 hours
                          (df_sorted['response_time'] > 0)]
    
    avg_response = responses.groupby('user')['response_time'].mean().sort_values()
    return avg_response

def message_length_analysis(selected_user, df):
    """Analyze message lengths"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['user'] != 'group_notification'].copy()
    media_patterns = ['image omitted', 'video omitted', 'audio omitted', 'sticker omitted', 'document omitted']
    temp = temp[~temp['message'].str.contains('|'.join(media_patterns), case=False, na=False)]
    
    temp['msg_length'] = temp['message'].apply(len)
    temp['word_count'] = temp['message'].apply(lambda x: len(x.split()))
    
    return temp

def conversation_starters(df):
    """Find who starts conversations most often"""
    df_sorted = df.sort_values('date').copy()
    df_sorted['time_diff'] = df_sorted['date'].diff().dt.total_seconds() / 3600  # in hours
    
    # Consider a new conversation if gap > 4 hours
    starters = df_sorted[df_sorted['time_diff'] > 4]['user'].value_counts()
    return starters

def hourly_activity(selected_user, df):
    """Get hourly message distribution"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    hourly = df.groupby('hour').count()['message'].reindex(range(24), fill_value=0)
    return hourly

def get_chat_summary(df):
    """Generate a summary of the chat"""
    total_messages = df.shape[0]
    total_users = df[df['user'] != 'group_notification']['user'].nunique()
    
    date_range = f"{df['date'].min().strftime('%d %b %Y')} to {df['date'].max().strftime('%d %b %Y')}"
    total_days = (df['date'].max() - df['date'].min()).days + 1
    
    media_patterns = ['image omitted', 'video omitted', 'audio omitted', 'sticker omitted', 'document omitted']
    total_media = df[df['message'].str.contains('|'.join(media_patterns), case=False, na=False)].shape[0]
    
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))
    
    return {
        'total_messages': total_messages,
        'total_users': total_users,
        'date_range': date_range,
        'total_days': total_days,
        'total_media': total_media,
        'total_links': len(links),
        'avg_messages_per_day': round(total_messages / max(total_days, 1), 1)
    }

def get_most_active_date(selected_user, df):
    """Find the most active date"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    most_active = df.groupby('only_date').count()['message'].idxmax()
    count = df.groupby('only_date').count()['message'].max()
    return most_active, count

def get_link_domains(df):
    """Extract and count domains from shared links"""
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))
    
    domains = []
    for link in links:
        try:
            # Extract domain from URL
            match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', link)
            if match:
                domains.append(match.group(1))
        except:
            continue
    
    domain_df = pd.DataFrame(Counter(domains).most_common(10), columns=['Domain', 'Count'])
    return domain_df

def night_owl_analysis(df):
    """Analyze who sends most messages during night (11 PM - 5 AM)"""
    night_df = df[(df['hour'] >= 23) | (df['hour'] <= 5)]
    night_users = night_df[night_df['user'] != 'group_notification']['user'].value_counts()
    return night_users

def get_streak_info(df):
    """Find the longest chat streak (consecutive days with messages)"""
    dates = df['only_date'].unique()
    dates = sorted(dates)
    
    if len(dates) < 2:
        return 1, dates[0] if dates else None, dates[0] if dates else None
    
    max_streak = 1
    current_streak = 1
    streak_start = dates[0]
    max_streak_start = dates[0]
    max_streak_end = dates[0]
    
    for i in range(1, len(dates)):
        diff = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i-1])).days
        if diff == 1:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
                max_streak_start = streak_start
                max_streak_end = dates[i]
        else:
            current_streak = 1
            streak_start = dates[i]
    
    return max_streak, max_streak_start, max_streak_end















