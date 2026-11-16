import streamlit as st
import pandas as pd
import plotly.express as px

# BASELINE FEATURES

# Set page config and title
st.set_page_config(page_title='Spotify Listening Analysis', layout='wide')
st.title('Spotify Listening History Analysis')

# functions

def time_of_day(x):
    if x.hour < 5:
        return 'Night'
    elif x.hour < 12:
        return 'Morning'
    elif x.hour  < 18:
        return 'Afternoon'
    elif x.hour < 21:
        return 'Evening'
    else:
        return 'Night'


def mood(val, ene):
    if val >= 0.5 and ene >=0.5:
        return 'Upbeat'
    elif val >= 0.5 and ene < 0.5:
        return 'Calm'
    elif val < 0.5 and ene >= 0.5:
        return 'Tense'
    elif val < 0.5 and ene < 0.5:
        return 'Mellow'
    else:
        return "Unknown"

@st.cache_data
def load_and_process_data():

    # load data
    audio_features = pd.read_json("audio_features.json", orient='index')
    spotify_tracks = pd.read_csv("spotify_tracks_data.csv")
    spotify_history = pd.read_csv("spotify_history.csv")

    # DATA PRE-PROCESSING:

    # remove duplicates from spotify tracks
    cols_no_track_genre = spotify_tracks.columns.drop('track_genre')
    spotify_tracks = spotify_tracks.drop_duplicates(subset=cols_no_track_genre, keep='first')

    # rename columns and convert to datetime for spotify history
    spotify_history = spotify_history.rename(columns={'spotify_track_uri':'track_id', 'artist_name':'artists'})
    spotify_history['ts'] = pd.to_datetime(spotify_history['ts'], errors='coerce')
    spotify_history = spotify_history.dropna(subset=['ts'])


    # filter for valid spotify tracks
    filter_features = audio_features.dropna(subset='range').index
    range_dict = {feature: audio_features.loc[feature]['range'] for feature in filter_features}

    valid_spotify_tracks = pd.Series(True, index=spotify_tracks.index)

    for feature in filter_features:
        valid_spotify_tracks &= spotify_tracks[feature].between(range_dict[feature][0], range_dict[feature][1])
        
    spotify_tracks = spotify_tracks[valid_spotify_tracks]


    # part 1: merge spotify_history with spotify_tracks
    same_feature = set(spotify_history.columns) & set(spotify_tracks.columns)
    same_feature.discard('track_id')
    spotify_tracks = spotify_tracks.drop(columns=same_feature)
    merged_spotify = pd.merge(spotify_history, spotify_tracks, how='left', on='track_id', indicator=True)

    #part 2: new columns transformation
    merged_spotify['matched'] = [1 if indicator == 'both' else 0 for indicator in merged_spotify['_merge']]
    merged_spotify['date'] = merged_spotify['ts'].dt.date
    merged_spotify['hour'] = merged_spotify['ts'].dt.hour
    merged_spotify['min_played'] = merged_spotify['ms_played']/60000
    merged_spotify['time_of_day'] = merged_spotify['ts'].apply(time_of_day)
    merged_spotify['mood'] = merged_spotify.apply(lambda row: mood(row['valence'], row['energy']), axis=1)
    merged_spotify['day_of_week'] = pd.to_datetime(merged_spotify['date']).dt.day_name() # add day of week for added feature 2: listening behavior heatmap
    merged_spotify = merged_spotify.drop(columns='_merge')

    return merged_spotify, audio_features

merged_spotify, audio_features = load_and_process_data()


# sidebar filter 
st.sidebar.header('Filters')

# BASELINE FILTERS: add date range, time of day (tod), skipped, and matched audio features to the side bar widget
date = st.sidebar.slider('Date', min(merged_spotify['date']), max(merged_spotify['date']), (min(merged_spotify['date']), max(merged_spotify['date'])))
tod = st.sidebar.multiselect('Time of Day', sorted(merged_spotify['time_of_day'].unique()), default = sorted(merged_spotify['time_of_day'].unique()))
skip = st.sidebar.checkbox('Exclude skipped tracks', value=False)
matched = st.sidebar.checkbox('Matched to audio features', value=False)

# EXPLORATORY ADDITION
# EXPLORATORY FEATURE 1 (SIDEBAR FILTER): Filter by artist
artists = st.sidebar.text_input('Search Artist', placeholder='Type part of an artist name')

# create filters
m_date = merged_spotify['date'].between(date[0], date[1])
m_tod = merged_spotify['time_of_day'].isin(tod)

view = merged_spotify[m_date & m_tod]

if skip:
    view = view[view['skipped'] != True]

if matched:
    view = view[view['matched'] == 1]

# Filter records to include only artists matching the search query
if artists.strip():
    view = view[view['artists'].str.contains(artists.strip(),case=False, na=False, regex=False)]
    
if view.empty:
    st.warning("No data matches these filters!")
    st.stop()


# create headline metrics
c1, c2, c3, c4 = st.columns(4)

c1.metric('Tracks Played', f'{len(view):,}')
c2.metric('Hours Played', round(view['min_played'].sum()/60))
c3.metric('Unique Tracks', view['track_id'].nunique())
c4.metric('Unique Artists', view['artists'].nunique())

# create multiple tabs for different visualization/charts
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Average Audio Profile', 'Tracks by Mood', 'Filtered Data', 'Top Favorites', 'Listening Behavior'])

float_features = audio_features[audio_features['type'] == 'float'].index

# BASELINE TAB 1: RADAR CHART OF AVERAGE AUDIO FEATURES, NORMALIZED TO [0,1]
with tab1:
    st.subheader('Average Audio Profile')

    features_means = view[float_features].mean()

    loudness_max, loudness_min = audio_features.loc['loudness']['range'][1], audio_features.loc['loudness']['range'][0]
    features_means['loudness'] = (features_means['loudness'] - loudness_min) / (loudness_max - loudness_min)

    tempo_max, tempo_min = merged_spotify['tempo'].max(), merged_spotify['tempo'].min()
    features_means.loc['tempo'] = (features_means.loc['tempo'] - tempo_min) /(tempo_max - tempo_min)

    df = pd.DataFrame(dict(
            r=features_means.values,
            theta=features_means.index))

    fig_audio_profile = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig_audio_profile.update_traces(fill='toself')

    st.plotly_chart(fig_audio_profile, width='stretch')

# BASELINE TAB 2: PIE CHART OF MOOD PROPORTIONS, EXCLUDING "UNKNOWN"
with tab2:
    st.subheader('Tracks by Mood')
    valid_mood = view[view['mood'] != 'Unknown']

    mood_counts = valid_mood['mood'].value_counts().reset_index()
    mood_counts.columns = ['Mood', 'Count']

    fig_mood_pie = px.pie(mood_counts, values='Count', names='Mood', title='Proportion of tracks played by mood')

    st.plotly_chart(fig_mood_pie, width='stretch')


# BASELINE TAB 3: FILTERED DATA TABLE AND DOWNLOAD CSV FEATURE
with tab3:
    st.subheader('Filtered Data')

    displayed_features = ['track_name', 'artists', 'album_name', 'track_genre', 'mood', 'min_played', 'time_of_day', 'ts', 'skipped'] + audio_features.index.to_list()

    st.dataframe(view[displayed_features], width='stretch')

    st.download_button(
        'Download filtered CSV',
        view[displayed_features].to_csv(index=False).encode('utf-8'),
        'filtered_data.csv',
        'text/csv'
    )



# EXPLORATORY ADDITIONS 

# EXPLORATORY FEATURE 2 (TAB 4): NEW TAB WITH A HORIZONTAL BAR CHART SHOWING THE TOP Nth ARTISTS AND TOP Nth TRACKS RANKED BY MINUTES PLAYED
with tab4:
    # Group data by artists, sum the minutes played, sort in descending order then select the first 10 
    top_artists = view.groupby('artists')['min_played'].sum().reset_index().sort_values(by='min_played', ascending=False)[:10]

    # Display the top nth favorites based on the number of top artists (max is 10)
    st.subheader(f'Top {len(top_artists)} Most Played Artists')
    
    # Create a horizontal barchart in descending order to show the top Nth most played artists
    fig_top_artists = px.bar(top_artists, x='min_played', y='artists',
                                orientation='h',
                                labels={
                                    'min_played': 'Minutes Played',
                                    'artists': 'Artist'
                                },
                                color='min_played',
                                color_continuous_scale='Viridis')
    fig_top_artists.update_layout(xaxis_title = 'Minutes Played', yaxis_title = 'Artist', 
                                  xaxis=dict(title_font=dict(size=20),
                                            tickfont=dict(size=14)),
                                yaxis=dict(categoryorder='total ascending',
                                            title_font=dict(size=20),
                                            tickfont=dict(size=14)))
    st.plotly_chart(fig_top_artists, width='stretch')



    
    # Group data by track_id, track_name and artists, sum the minutes played, sort in descending order then select first 10
    top_track = (view.groupby(['track_id', 'track_name', 'artists'])['min_played'].sum().reset_index().sort_values(by='min_played', ascending=False)[:10])

    # Display the top nth most played track based on filter (max is 10)
    st.subheader(f'Top {len(top_track)} Most Played Tracks')
    
    top_track['label'] = top_track['track_name'] + ' - ' + top_track['artists']

    # Create a horizontal bar chart in descending order to show the top 10 most played tracks
    fig_top_track = px.bar(top_track, x='min_played', y='label',
                                orientation='h',
                                labels={
                                    'min_played': 'Minutes Played',
                                    'label': 'Track'
                                },
                                color='min_played',
                                color_continuous_scale='Viridis')
    fig_top_track.update_layout(xaxis_title = 'Minutes Played', yaxis_title = 'Track', 
                                    xaxis=dict(title_font=dict(size=20),
                                            tickfont=dict(size=14)),
                                yaxis=dict(categoryorder='total ascending',
                                            title_font=dict(size=20),
                                            tickfont=dict(size=14)))
    st.plotly_chart(fig_top_track, width='stretch')

# EXPLORATORY FEATURE 3 (TAB 5): HEATMAP OF LISTENING BEHAVIOR MEASURED BY MINUTES PLAYED ACROSS DAY OF THE WEEK AND TIME OF THE DAY
with tab5:

    st.subheader('Listening Behavior')

    # Group by day of week and time of day, sum the minutes played
    behavior_data = view.groupby(['day_of_week', 'time_of_day'])['min_played'].sum().reset_index()

    # Convert to matrix format
    heatmap_data = behavior_data.pivot(index='day_of_week', columns='time_of_day', values='min_played')

    # Define the order for days and times 
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    times = ['Morning', 'Afternoon', 'Evening', 'Night']

    # Reorder according to the order above
    heatmap_data = heatmap_data.reindex(index=days, columns=times).fillna(0)

    # Create the heatmap where the color intensity corresponds to the minutes played 
    fig_heatmap_behavior = px.imshow(heatmap_data, labels = dict(x='Time of Day', y='Day of the Week', color='Minutes played'),
                                     color_continuous_scale='Viridis', text_auto='.0f', aspect='auto', title='Listening Activity by Day and Time')

    fig_heatmap_behavior.update_layout(xaxis=dict(title_font=dict(size=20), tickfont=dict(size=15)),
                                       yaxis = dict(title_font=dict(size=20), tickfont=dict(size=15)))

    st.plotly_chart(fig_heatmap_behavior, width='stretch')

    
