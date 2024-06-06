import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import Counter
from nltk import tokenize, corpus, stem
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#https://raw.githubusercontent.com/jneidel/job-titles/master/job-titles.json

# Konfigurasi awal streamlit
st.set_page_config(
    page_title = '', 
    page_icon = '💰', 
    layout = 'wide'
)

import nltk
nltk.download('stopwords')

#@st.cache_data
def extract_data():
    lowongan_kerja = pd.read_excel(
        'https://raw.githubusercontent.com/Rizkiramdani04/byte_brigade/main/cleaned_lowongan_kerja_only.xlsx',
        dtype = {
            'id_pekerjaan' : pd.Int64Dtype(),
            'id_perusahaan' : pd.Int64Dtype()
        }
    )
    
    return(lowongan_kerja)

def remove_stopwords(text):
    stop_words = set(corpus.stopwords.words('english'))
    lemmatizer = stem.WordNetLemmatizer()
    
    tokens = tokenize.word_tokenize(text)
    filtered_text = [word for word in tokens if word not in stop_words]
    filtered_text_lemmatized = [lemmatizer.lemmatize(word) for word in filtered_text]
    filtered_text_lemmatized = ' '.join(set(filtered_text_lemmatized))
    return (filtered_text_lemmatized)

#@st.cache_data
def transform_data(data):

    # Bersihkan data judul
    judul_pat_delete_regex = [
        '\$\d+(/hr)?\s*\|?\s*', 
        '\s?\([^)]*\)?', 
        '\$+[\s\d,]', 
        ' in .+',
        '[^a-zA-Z\s]'
    ]
    
    judul_clean = data['job_title_group'].str.upper()
    for pat in judul_pat_delete_regex:
        judul_clean = judul_clean.str.replace(pat, '', regex = True)
    
    judul_clean = judul_clean.str.strip()
    
    # Bersihkan data deskripsi
    regex_replace = {
        '\n' : ' ',
        '\s+' : ' ',
        '[^a-z0-9\s]' : ''
    }
    
    deskripsi_clean = data['deskripsi'].str.lower()
    deskripsi_clean = deskripsi_clean.fillna('')
    
    for pat, repl in regex_replace.items():
        deskripsi_clean = deskripsi_clean.replace(pat, repl, regex = True)
    
    deskripsi_clean = deskripsi_clean.apply(remove_stopwords)
    
    # Bersihkan data jenis_pekerjaan
    jenis_pekerjaan = data['jenis_pekerjaan_terformat'].str.replace('-', ' ', regex = False)
    jenis_pekerjaan = jenis_pekerjaan.str.title()
    
    data['job_title_group'] = judul_clean
    data['deskripsi'] =  deskripsi_clean
    data['jenis_pekerjaan_terformat'] = jenis_pekerjaan
    
    return(data)

#@st.cache_data
def get_dimension(
        data : pd.DataFrame, 
        column : str,
        add_all : bool
) -> list:
    
    data[column] = data[column].dropna()
    data[column] = data[column].str.upper()
    dim = data[column].sort_values().unique()
    dim = dim.tolist()
    
    if(add_all):
        dim = ['ALL'] + dim
    
    return(dim)
    
#@st.cache_data
def get_master_states(url_geojson):

    # Fetch the GeoJSON data
    url_geojson = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json'
    response = requests.get(url_geojson)
    geojson_data = response.json()
    fitur_geo = geojson_data.get('features')
    
    state_id = list()
    state_name = list()
    
    for fitur in fitur_geo:
        state_id.append(fitur.get('id'))
        state_name.append(fitur.get('properties')['name'])
    
    state_centers = {
        'AL': [32.806671, -86.791130],
        'AK': [61.370716, -152.404419],
        'AZ': [33.729759, -111.431221],
        'AR': [34.969704, -92.373123],
        'CA': [36.116203, -119.681564],
        'CO': [39.059811, -105.311104],
        'CT': [41.597782, -72.755371],
        'DE': [39.318523, -75.507141],
        'FL': [27.766279, -81.686783],
        'GA': [33.040619, -83.643074],
        'HI': [21.094318, -157.498337],
        'ID': [44.240459, -114.478828],
        'IL': [40.349457, -88.986137],
        'IN': [39.849426, -86.258278],
        'IA': [42.011539, -93.210526],
        'KS': [38.526600, -96.726486],
        'KY': [37.668140, -84.670067],
        'LA': [31.169546, -91.867805],
        'ME': [44.693947, -69.381927],
        'MD': [39.063946, -76.802101],
        'MA': [42.230171, -71.530106],
        'MI': [43.326618, -84.536095],
        'MN': [45.694454, -93.900192],
        'MS': [32.741646, -89.678696],
        'MO': [38.456085, -92.288368],
        'MT': [46.921925, -109.354353],
        'NE': [41.125370, -98.268082],
        'NV': [38.313515, -117.055374],
        'NH': [43.452492, -71.563896],
        'NJ': [40.298904, -74.521011],
        'NM': [34.840515, -106.248482],
        'NY': [42.165726, -74.948051],
        'NC': [35.630066, -79.806419],
        'ND': [47.528912, -99.784012],
        'OH': [40.388783, -82.764915],
        'OK': [35.565342, -96.928917],
        'OR': [44.572021, -122.070938],
        'PA': [40.590752, -77.209755],
        'RI': [41.680893, -71.511780],
        'SC': [33.856892, -80.945007],
        'SD': [44.299782, -99.438828],
        'TN': [35.747845, -86.692345],
        'TX': [31.054487, -97.563461],
        'UT': [40.150032, -111.862434],
        'VT': [44.045876, -72.710686],
        'VA': [37.769337, -78.169968],
        'WA': [47.400902, -121.490494],
        'WV': [38.491226, -80.954456],
        'WI': [44.268543, -89.616508],
        'WY': [42.755966, -107.302490]
    }
    
    master_state = pd.DataFrame({
        'state_id' : state_id,
        'state_name' : state_name,
        'lat': [state_centers[abbr][0] for abbr in state_id],
        'lon': [state_centers[abbr][1] for abbr in state_id]
    })
    
    return(master_state)

def usa_map(data, select_judul_loker):
    
    url_geojson = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json'
    master_state = get_master_states(url_geojson)
    
    data_group = data.groupby('state_id', as_index = False).agg(values = ('id_pekerjaan', 'count'))
    
    data_group = data_group.merge(
        master_state, 
        on = 'state_id', 
        how = 'right'
    )
    
    data_group['values'] = data_group['values'].fillna(0)
    
    fig = go.Figure(
        go.Choropleth(
            geojson = url_geojson,
            locations = data_group['state_id'],
            z = data_group['values'],
            colorscale = 'Blues',
            text = data_group['state_name'],
            hoverinfo = 'text+z',
            hovertemplate = select_judul_loker.title() + ' in <b>%{text}</b><br>Jumlah Lowongan : %{z}'
        )
    )
    
    fig.update_traces(
        showscale = False
    )
    
    for i, row in data_group.iterrows():
        fig.add_trace(
            go.Scattergeo(
              locationmode = 'USA-states',
              lon = [row['lon']],
              lat = [row['lat']],
              text = row['state_id'],
              mode = 'text',
              showlegend = False,
              hoverinfo = 'text'
        ))
    
    fig.update_layout(
        height = 375,
        width = 600,
        geo = dict(
            scope = 'usa',
        ),
        margin = dict(
            l = 50,
            r = 0,
            t = 0,
            b = 45
        )
    )

    return (fig)

def text_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]


def layout(data_input):
    data = data_input.copy()
    
    _, row0, _ = st.columns([0.1, 18, 0.1])
    _, row2, _, row1, _ = st.columns([0.1, 4, 0.1, 4, 0.1])
    _, row3, _ = st.columns([0.1, 18, 0.1])
    
    row0.title('Compare Your Summary to a Job Description!')
    row0.markdown('''
        Alat Pencocokan Deskripsi Pekerjaan dalam Resume memungkinkan\
        untuk dengan cepat membandingkan resume Anda yang sudah ada dengan\
        deskripsi pekerjaan dari peran apa pun. Dapatkan skor pencocokan instan\
        dengan rincian seberapa baik resume Anda sejalan dengan bahasa,\
        kata kunci, dan keterampilan dari pekerjaan tersebut.\
        Lihat bagaimana resume Anda dibandingkan dengan setiap pekerjaan.'''
    .strip())
    
    dim_judul_loker = get_dimension(data = data, column = 'job_title_group', add_all = False)
    dim_negara_bagian_perusahaan = get_dimension(data = data, column = 'state_name', add_all = True)
    dim_jenis_loker = get_dimension(data = data, column = 'jenis_pekerjaan_terformat', add_all = True)
    dim_tingkat_pengalaman = get_dimension(data = data, column = 'tingkat_pengalaman_terformat', add_all = True)
    
    select_judul_loker = row2.selectbox(label = 'Judul Loker', options = dim_judul_loker, index = dim_judul_loker.index('DATA SCIENTIST'))
    select_lokasi_perusahaan = row2.selectbox(label = 'Lokasi', options = dim_negara_bagian_perusahaan)
    select_jenis_loker = row2.selectbox(label = 'Jenis Pekerjaan', options = dim_jenis_loker)
    select_pengalaman_loker = row2.selectbox(label = 'Tingkat', options = dim_tingkat_pengalaman)
    
    condition1 = data['job_title_group'].str.contains(select_judul_loker, na = False, case = False)
    
    if((select_lokasi_perusahaan != 'ALL')):
        condition2 = (data['state_name'].str.contains(select_lokasi_perusahaan, na = False, case = False))
        if(data[condition1 & condition2].shape[0] == 0):
            row3.warning(f'Pencarian Lowongan {select_judul_loker} di {select_lokasi_perusahaan} tidak ditemukan!\nLowongan yang mungkin anda cari :')
            data = data[condition1]
        else:
            data = data[condition1 & condition2]
    else:
        data = data[condition1]
    
    hitung_error = 0;
    if((select_jenis_loker != 'ALL')):
        condition3 = (data['jenis_pekerjaan_terformat'].str.contains(select_jenis_loker, na = False, case = False))
        if(data[condition3].shape[0] == 0):
            row3.warning(f'Pencarian Lowongan {select_judul_loker} di {select_jenis_loker} tidak ditemukan!\nLowongan yang mungkin anda cari :')
        else:
            data = data[condition3]

    if((select_pengalaman_loker != 'ALL')):
        condition4 = (data['tingkat_pengalaman_terformat'].str.contains(select_pengalaman_loker, na = False, case = False))
        if(data[condition4].shape[0] == 0):
            row3.warning(f'Pencarian Lowongan {select_judul_loker} di {select_pengalaman_loker} tidak ditemukan!\nLowongan yang mungkin anda cari :')
        else:
            data = data[condition4]
    
    fig = usa_map(data, select_judul_loker)
    row1.plotly_chart(figure_or_data = fig, use_container_width = False)
    
    row3.subheader('Check Your Summary')
    summary = row3.text_input(label = 'Input Summary')
    
    if(summary != ''):
        list_data = data.to_dict('records')
        for data_dict in list_data:
            description = data_dict.get('deskripsi')
            match_score = text_similarity(summary, description)
            match_score = round(match_score, 4) * 100
            match_score = "{:.2f}".format(match_score)
            
            data_dict.update({
                'match_score' : match_score
            })
            
        data_match_score = pd.DataFrame(list_data)
        data_match_score['match_score'] = data_match_score['match_score'].astype(pd.Float64Dtype())
        data_match_score = data_match_score.sort_values(by = ['match_score'], ascending = False, ignore_index = True)
        data_match_score_dict = data_match_score.to_dict('records')
        
        _, row4, _, row5, _ = st.columns([0.1, 3, 15, 5, 0.1])
        _, row6, _ = st.columns([0.1, 18, 0.1])
        
        row4.subheader('Result')
        limit_show = row6.slider(label = 'Show', min_value = 1, max_value = len(data_match_score_dict), value = 1)
        
        deskripsi_full = str()
        for data_dict in data_match_score_dict[:limit_show]:
            job_title = data_dict.get('job_title_group')
            company = data_dict.get('nama_perusahaan')
            experience = data_dict.get('tingkat_pengalaman_terformat')
            states = data_dict.get('state_name')
            match_score = data_dict.get('match_score')
            deskripsi = data_dict.get('deskripsi')
            jenis_pekerjaan = data_dict.get('jenis_pekerjaan_terformat')
            gaji_min = int(round(data_dict.get('gaji_monthly_min'), 0))
            gaji_max = int(round(data_dict.get('gaji_monthly_max'), 0))
            
            deskripsi_full += deskripsi 
            # atas-kanan-bawah-kiri,
            row6.markdown(f'''
                <div style="
                     border: 1px solid #2B35AF; 
                     display: flex; 
                     justify-content: space-between; 
                     border-radius: 20px;
                     align-items: center;
                     background-color:#D9F1FF;">
                     <div style="width: 80%">
                         <div style="font-size: 40px; padding: 10px 0px 0px 20px;"><b>{job_title}</b></div>
                         <div style="margin: -0.5em 0 0 0; font-size: 25px; padding: 0px 0px 0px 20px;"><i>{company}, {states}</i></div>
                         <div style="margin: 0 -2em; padding: 10px 0px 10px 70px;">
                             <ul>
                                 <li style="list-style: none; font-size: 15px;">💼  {experience}</li>
                                 <li style="list-style: none; font-size: 15px;">📋  {jenis_pekerjaan}</li>
                                 <li style="list-style: none; font-size: 15px;">💵  ${gaji_min} - ${gaji_max} (Monthly)</li>
                             </ul>
                         </div>
                      </div>
                      <div style="width: 20%; text-align: center;">
                        <div style="font-size: 70px;padding: 10px 0px 3px 20px;">
                            <b>{match_score}%</b>
                        </div>
                        <div style="font-size: 20px; text-align: center;padding: 0px 0px 10px 20px;">Matched Score</div>
                      </div>
                    </div>           
            '''.strip(), unsafe_allow_html = True)
            row6.markdown('<br>', unsafe_allow_html = True)
        
        row6.subheader('Word Cloud')
        row6.markdown(f'Gunakan kata - kata berikut untuk meningkatkan matching score dengan pekerjaan yang sedang dicari : {select_judul_loker}')
        fig_wc = wordcloud(deskripsi_full)
        row6.pyplot(fig = fig_wc, use_container_width=False)


def wordcloud(text):
    word_counts = Counter(text.split())
    
    wordcloud = WordCloud(
        width = 400, 
        height = 200, 
        background_color = 'white', 
        stopwords = STOPWORDS,
        colormap = 'Blues',
        prefer_horizontal = 1.0
    ).generate(text)
    
    # Create the figure
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    return fig

if __name__ == '__main__':
    
    data_loker = extract_data()
    data_loker_transformed = transform_data(data_loker)
    
    layout(data_loker_transformed)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
