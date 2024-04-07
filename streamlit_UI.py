import pandas as pd 
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import os
from PIL import Image

#importing the csv file to encode catgorical values
df = pd.read_csv("/Users/velmurugan/Desktop/velu/python_works/singapore_flat_price_prediction/processed_singapore_flat.csv")

#encoding the town values like label encoding 
town_uni = sorted(list(df.town.unique()))

town_dict = {}

for ind,town in enumerate(town_uni):
    town_dict[town] = int(ind)

def town_mapping(town):
    town_encode = town_dict[town]
    return town_encode

print(town_mapping("ANG MO KIO"))


#encoding the flat type values
flat_type_uni = sorted(list(df.flat_type.unique()))

flat_type_dict = {}

for ind,flat_type in enumerate(flat_type_uni):
    flat_type_dict[flat_type] = ind

def flat_type_mapping(flat_type):
    flat_type_encode = flat_type_dict[flat_type]
    return flat_type_encode

print(flat_type_mapping("1 ROOM"))

#encoding the street_name values

street_name_unique = sorted(list(df.street_name.unique()))

street_name_dict ={}

for ind,street_name in enumerate(street_name_unique):
    street_name_dict[street_name]=ind

def street_name_mapping(street_name):
    street_name_encode = street_name_dict[street_name]
    return street_name_encode

print(street_name_mapping("ANG MO KIO AVE 1"))


#encoding the flat_model values

flat_model_unique = sorted(list(df.flat_model.unique()))

flat_model_dict = {}

for ind,flat_model in enumerate(flat_model_unique):
    flat_model_dict[flat_model] = ind

def flat_model_mapping(flat_model):
    flat_model_encode = flat_model_dict[flat_model]
    return flat_model_encode

print(flat_model_mapping("Improved"))


#function for predicting the flat resale_price

def predict_resale_price(town,flat_type,block,street_name,floor_area_sqm,
                        flat_model,lease_commence_date,storey_range_start,
                        storey_range_end,resale_year,resale_month):
    town = town_mapping(town)
    flat_type = flat_type_mapping(flat_type)
    block = float(block)
    street_name = street_name_mapping(street_name)
    floor_area_sqm = float(floor_area_sqm)
    flat_model = flat_model_mapping(flat_model)
    lease_commence_date = int(lease_commence_date)
    storey_range_start = np.log(storey_range_start)
    storey_range_end = np.log(storey_range_end)
    resale_year = int(resale_year)
    resale_month = int(resale_month) 

    user_feed = np.array([[town,flat_type,block,street_name,floor_area_sqm,
                            flat_model,lease_commence_date,storey_range_start,
                            storey_range_end,resale_year,resale_month]]) 
    print(user_feed)
    
    # loading the trained model
    file_path = "/Users/velmurugan/Desktop/velu/python_works/singapore_flat_price_prediction/random_forest_model.pkl"
    
    with open(file_path,"rb") as f:
        regg_model = pickle.load(f)
    
    y_pred = regg_model.predict(user_feed)
    y_pred_actual = np.exp(y_pred[0])
    pred_price = round(y_pred_actual)

    return pred_price


#streamlit part code starts here

st.set_page_config(layout="wide")

st.title("SINGAPORE RESALE FLAT PRICES PREDICTING")
st.write("")

with st.sidebar:
    select= option_menu("MAIN MENU",["Home", "Price Prediction", "About"])

if select == "Home":
    img= Image.open("/Users/velmurugan/Desktop/velu/python_works/singapore_flat_price_prediction/flat_pred.jpg")
    st.image(img,500,500)

    st.write("Our project aims to develop a machine learning model capable of accurately predicting resale flat prices in Singapore. Housing affordability and market transparency are critical concerns for both buyers and sellers in Singapore's real estate market. By leveraging historical transaction data and advanced machine learning techniques, our model provides valuable insights into future resale flat prices, aiding buyers, sellers, and real estate professionals in making informed decisions.")

    

elif select == "Price Prediction":

    col1,col2= st.columns(2)
    with col1:

        resale_year= st.selectbox("Select Resale Year",[1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
                                                    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                                    2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,
                                                    2023, 2024])


        resale_month= st.selectbox("Select Resale Month",[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])                                         
                                                    
        lease_commence_date= st.selectbox("Select Lease Commence Year",[1977, 1976, 1978, 1979, 1984, 1980, 1985, 1981, 1982, 1986, 1972,
                                                    1983, 1973, 1969, 1975, 1971, 1974, 1967, 1970, 1968, 1988, 1987,
                                                    1989, 1990, 1992, 1993, 1994, 1991, 1995, 1996, 1997, 1998, 1999,
                                                    2000, 2001, 1966, 2002, 2006, 2003, 2005, 2004, 2008, 2007, 2009,
                                                    2010, 2012, 2011, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022,
                                                    2020])
        
        town= st.selectbox("Select Town", ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                                            'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                                            'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                                            'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                                            'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
                                            'TOA PAYOH', 'WOODLANDS', 'YISHUN'])
        
        flat_type= st.selectbox("Select Flat Type", ['3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', '1 ROOM',
                                                        'MULTI-GENERATION'])
        
        

        flat_model= st.selectbox("Select Flat Model", ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
                                                        'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
                                                        'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
                                                        'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
                                                        'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen'])
        
    with col2:

        floor_area_sqm= st.number_input("Enter  Floor Area(sqm) (Min: 31 / Max: 280")

        storey_range_start= st.number_input("Enter Starting number of  Storey")

        storey_range_end= st.number_input("Enter ending number of storey")

        block = st.slider('Select Block:', 1, 980)

        street_name = st.selectbox("Select Street", ['ADMIRALTY DR', 'ADMIRALTY LINK', 'AH HOOD RD', 'ALEXANDRA RD', 'ALJUNIED AVE 2', 'ALJUNIED CRES', 'ALJUNIED RD', 
                                                     'ANCHORVALE CRES', 'ANCHORVALE DR', 'ANCHORVALE LANE', 'ANCHORVALE LINK', 'ANCHORVALE RD', 'ANCHORVALE ST', 
                                                     'ANG MO KIO AVE 1', 'ANG MO KIO AVE 10', 'ANG MO KIO AVE 2', 'ANG MO KIO AVE 3', 'ANG MO KIO AVE 4', 'ANG MO KIO AVE 5',
                                                    'ANG MO KIO AVE 6', 'ANG MO KIO AVE 8', 'ANG MO KIO AVE 9', 'ANG MO KIO ST 11', 'ANG MO KIO ST 21', 'ANG MO KIO ST 31', 'ANG MO KIO ST 32', 'ANG MO KIO ST 44', 'ANG MO KIO ST 51', 'ANG MO KIO ST 52', 'ANG MO KIO ST 61', 'BAIN ST', 'BALAM RD', 'BANGKIT RD', 'BEACH RD', 'BEDOK CTRL', 'BEDOK NTH AVE 1', 'BEDOK NTH AVE 2', 'BEDOK NTH AVE 3', 'BEDOK NTH AVE 4', 'BEDOK NTH RD', 'BEDOK NTH ST 1', 'BEDOK NTH ST 2',
                                                    'BEDOK NTH ST 3', 'BEDOK NTH ST 4', 'BEDOK RESERVOIR CRES', 'BEDOK RESERVOIR RD', 'BEDOK RESERVOIR VIEW', 'BEDOK STH AVE 1', 'BEDOK STH AVE 2', 'BEDOK STH AVE 3', 'BEDOK STH RD', 'BENDEMEER RD', 'BEO CRES', 'BISHAN ST 11', 'BISHAN ST 12', 'BISHAN ST 13', 'BISHAN ST 22', 'BISHAN ST 23', 'BISHAN ST 24', 'BOON KENG RD', 'BOON LAY AVE', 'BOON LAY DR', 'BOON LAY PL', 'BOON TIONG RD', 'BRIGHT HILL DR', 'BT BATOK CTRL',
                                                    'BT BATOK EAST AVE 3', 'BT BATOK EAST AVE 4', 'BT BATOK EAST AVE 5', 'BT BATOK EAST AVE 6', 'BT BATOK ST 11', 'BT BATOK ST 21', 'BT BATOK ST 22', 'BT BATOK ST 24', 'BT BATOK ST 25', 'BT BATOK ST 31', 'BT BATOK ST 32', 'BT BATOK ST 33', 'BT BATOK ST 34', 'BT BATOK ST 51', 'BT BATOK ST 52', 'BT BATOK WEST AVE 2', 'BT BATOK WEST AVE 4', 'BT BATOK WEST AVE 5', 'BT BATOK WEST AVE 6', 'BT BATOK WEST AVE 7', 
                                                    'BT BATOK WEST AVE 8', 'BT BATOK WEST AVE 9', 'BT MERAH CTRL', 'BT MERAH LANE 1', 'BT MERAH VIEW', 'BT PANJANG RING RD', 'BT PURMEI RD', 'BUANGKOK CRES', 'BUANGKOK GREEN', 'BUANGKOK LINK', 'BUANGKOK STH FARMWAY 1', 'BUFFALO RD', "C'WEALTH AVE", "C'WEALTH AVE WEST", "C'WEALTH CL", "C'WEALTH CRES", "C'WEALTH DR", 'CAMBRIDGE RD', 'CANBERRA CRES', 'CANBERRA LINK', 'CANBERRA RD', 'CANBERRA ST', 'CANBERRA WALK', 'CANTONMENT CL', 
                                                    'CANTONMENT RD', 'CASHEW RD', 'CASSIA CRES', 'CHAI CHEE AVE', 'CHAI CHEE DR', 'CHAI CHEE RD', 'CHAI CHEE ST', 'CHANDER RD', 'CHANGI VILLAGE RD', 'CHIN SWEE RD', 'CHOA CHU KANG AVE 1', 'CHOA CHU KANG AVE 2', 'CHOA CHU KANG AVE 3', 'CHOA CHU KANG AVE 4', 'CHOA CHU KANG AVE 5', 'CHOA CHU KANG AVE 7', 'CHOA CHU KANG CRES', 'CHOA CHU KANG CTRL', 'CHOA CHU KANG DR', 'CHOA CHU KANG LOOP', 'CHOA CHU KANG NTH 5', 'CHOA CHU KANG NTH 6', 'CHOA CHU KANG NTH 7', 'CHOA CHU KANG ST 51', 'CHOA CHU KANG ST 52', 'CHOA CHU KANG ST 53', 'CHOA CHU KANG ST 54', 'CHOA CHU KANG ST 62', 'CHOA CHU KANG ST 64', 'CIRCUIT RD', 'CLARENCE LANE', 'CLEMENTI AVE 1', 'CLEMENTI AVE 2', 'CLEMENTI AVE 3', 'CLEMENTI AVE 4', 'CLEMENTI AVE 5', 'CLEMENTI AVE 6', 'CLEMENTI ST 11', 'CLEMENTI ST 12', 'CLEMENTI ST 13', 'CLEMENTI ST 14', 'CLEMENTI WEST ST 1', 'CLEMENTI WEST ST 2', 'COMPASSVALE BOW', 'COMPASSVALE CRES', 'COMPASSVALE DR', 'COMPASSVALE LANE', 'COMPASSVALE LINK', 'COMPASSVALE RD', 'COMPASSVALE ST', 'COMPASSVALE WALK', 'CORPORATION DR', 'CRAWFORD LANE', 
                                                    'DAKOTA CRES', 'DAWSON RD', 'DELTA AVE', 'DEPOT RD', 'DORSET RD', 'DOVER CL EAST', 'DOVER CRES', 'DOVER RD', 'EAST COAST RD', 'EDGEDALE PLAINS', 'EDGEFIELD PLAINS', 'ELIAS RD', 'EMPRESS RD', 'EUNOS CRES', 'EUNOS RD 5', 'EVERTON PK', 'FAJAR RD', 'FARRER PK RD', 'FARRER RD', 'FERNVALE LANE', 'FERNVALE LINK', 'FERNVALE RD', 'FERNVALE ST', 'FRENCH RD', 'GANGSA RD', 'GEYLANG BAHRU', 'GEYLANG EAST AVE 1', 'GEYLANG EAST AVE 2', 'GEYLANG EAST CTRL', 'GEYLANG SERAI', 'GHIM MOH LINK', 'GHIM MOH RD', 'GLOUCESTER RD', 'HAIG RD', 'HAVELOCK RD', 'HENDERSON CRES', 'HENDERSON RD', 'HILLVIEW AVE', 'HO CHING RD', 'HOLLAND AVE', 'HOLLAND CL', 'HOLLAND DR', 'HOUGANG AVE 1', 'HOUGANG AVE 10', 'HOUGANG AVE 2', 'HOUGANG AVE 3', 'HOUGANG AVE 4', 'HOUGANG AVE 5', 'HOUGANG AVE 6', 'HOUGANG AVE 7', 'HOUGANG AVE 8', 'HOUGANG AVE 9', 'HOUGANG CTRL', 'HOUGANG ST 11', 'HOUGANG ST 21', 'HOUGANG ST 22', 'HOUGANG ST 31', 'HOUGANG ST 32', 'HOUGANG ST 51', 'HOUGANG ST 52', 'HOUGANG ST 61', 'HOUGANG ST 91', 'HOUGANG ST 92', 'HOY FATT RD', 'HU CHING RD', 'INDUS RD', 'JELAPANG RD', 'JELEBU RD', 'JELLICOE RD', 'JLN BAHAGIA', 'JLN BATU', 'JLN BERSEH', 'JLN BT HO SWEE', 'JLN BT MERAH', 'JLN DAMAI', 'JLN DUA', 'JLN DUSUN', 'JLN KAYU', 'JLN KLINIK', 'JLN KUKOH', "JLN MA'MOR", 'JLN MEMBINA', 'JLN MEMBINA BARAT', 'JLN PASAR BARU', 'JLN RAJAH', 'JLN RUMAH TINGGI', 'JLN TECK WHYE', 'JLN TENAGA', 'JLN TENTERAM', 'JLN TIGA', 'JOO CHIAT RD', 'JOO SENG RD', 'JURONG EAST AVE 1', 'JURONG EAST ST 13',
                                                    'JURONG EAST ST 21', 'JURONG EAST ST 24', 'JURONG EAST ST 31', 'JURONG EAST ST 32', 'JURONG WEST AVE 1', 'JURONG WEST AVE 3', 'JURONG WEST AVE 5', 'JURONG WEST CTRL 1', 'JURONG WEST CTRL 3', 'JURONG WEST ST 24', 'JURONG WEST ST 25', 'JURONG WEST ST 41', 'JURONG WEST ST 42', 'JURONG WEST ST 51', 'JURONG WEST ST 52', 'JURONG WEST ST 61', 'JURONG WEST ST 62', 'JURONG WEST ST 64', 'JURONG WEST ST 65', 'JURONG WEST ST 71', 'JURONG WEST ST 72', 'JURONG WEST ST 73', 'JURONG WEST ST 74', 'JURONG WEST ST 75', 'JURONG WEST ST 81', 'JURONG WEST ST 91', 'JURONG WEST ST 92', 'JURONG WEST ST 93', 'KALLANG BAHRU', 'KANG CHING RD', 'KEAT HONG CL', 'KEAT HONG LINK', 'KELANTAN RD', 'KENT RD', 'KG ARANG RD', 'KG BAHRU HILL', 'KG KAYU RD', 'KIM CHENG ST', 'KIM KEAT AVE', 'KIM KEAT LINK', 'KIM PONG RD', 'KIM TIAN PL', 'KIM TIAN RD', "KING GEORGE'S AVE", 'KLANG LANE', 'KRETA AYER RD', 'LENGKOK BAHRU', 'LENGKONG TIGA', 'LIM CHU KANG RD', 'LIM LIAK ST', 'LOMPANG RD', 'LOR 1 TOA PAYOH', 'LOR 1A TOA PAYOH', 'LOR 2 TOA PAYOH', 'LOR 3 GEYLANG', 'LOR 3 TOA PAYOH', 'LOR 4 TOA PAYOH', 'LOR 5 TOA PAYOH', 'LOR 6 TOA PAYOH', 'LOR 7 TOA PAYOH', 'LOR 8 TOA PAYOH', 'LOR AH SOO', 'LOR LEW LIAN', 'LOR LIMAU', 'LOWER DELTA RD', 'MACPHERSON LANE', 'MARGARET DR', 'MARINE CRES', 'MARINE DR', 'MARINE PARADE CTRL', 'MARINE TER', 'MARSILING CRES', 'MARSILING DR', 'MARSILING LANE', 'MARSILING RD', 'MARSILING RISE', 'MCNAIR RD', 'MEI LING ST', 'MOH GUAN TER', 'MONTREAL DR', 'MONTREAL LINK', 'MOULMEIN RD',
                                                    'NEW MKT RD', 'NEW UPP CHANGI RD', 'NILE RD', 'NTH BRIDGE RD', 'OLD AIRPORT RD', 'OUTRAM HILL', 'OUTRAM PK', 'OWEN RD', 'PANDAN GDNS', 'PASIR RIS DR 1', 'PASIR RIS DR 10', 'PASIR RIS DR 3', 'PASIR RIS DR 4', 'PASIR RIS DR 6', 'PASIR RIS ST 11', 'PASIR RIS ST 12', 'PASIR RIS ST 13', 'PASIR RIS ST 21', 'PASIR RIS ST 41', 'PASIR RIS ST 51', 'PASIR RIS ST 52', 'PASIR RIS ST 53', 'PASIR RIS ST 71', 'PASIR RIS ST 72', 'PAYA LEBAR WAY', 'PENDING RD', 'PETIR RD', 'PINE CL', 'PIPIT RD', 'POTONG PASIR AVE 1', 'POTONG PASIR AVE 2', 'POTONG PASIR AVE 3', 'PUNGGOL CTRL', 'PUNGGOL DR', 'PUNGGOL EAST', 'PUNGGOL FIELD', 'PUNGGOL FIELD WALK', 'PUNGGOL PL', 'PUNGGOL RD', 'PUNGGOL WALK', 'PUNGGOL WAY', 'QUEEN ST', "QUEEN'S CL", "QUEEN'S RD", 'QUEENSWAY', 'RACE COURSE RD', 'REDHILL CL', 'REDHILL LANE', 'REDHILL RD', 'RIVERVALE CRES', 'RIVERVALE DR', 'RIVERVALE ST', 'RIVERVALE WALK', 'ROCHOR RD', 'ROWELL RD', 'SAGO LANE', 'SAUJANA RD', 'SEGAR RD', 'SELEGIE RD', 'SELETAR WEST FARMWAY 6', 'SEMBAWANG CL', 'SEMBAWANG CRES', 'SEMBAWANG DR', 'SEMBAWANG RD', 'SEMBAWANG VISTA', 'SEMBAWANG WAY', 'SENG POH RD', 'SENGKANG CTRL', 'SENGKANG EAST AVE', 'SENGKANG EAST RD', 'SENGKANG EAST WAY', 'SENGKANG WEST AVE', 'SENGKANG WEST WAY', 'SENJA LINK', 'SENJA RD', 'SERANGOON AVE 1', 'SERANGOON AVE 2', 'SERANGOON AVE 3', 'SERANGOON AVE 4', 'SERANGOON CTRL', 'SERANGOON CTRL DR', 'SERANGOON NTH AVE 1', 'SERANGOON NTH AVE 2', 'SERANGOON NTH AVE 3', 'SERANGOON NTH AVE 4', 'SHORT ST', 'SHUNFU RD', 'SILAT AVE',
                                                    'SIMEI LANE', 'SIMEI RD', 'SIMEI ST 1', 'SIMEI ST 2', 'SIMEI ST 4', 'SIMEI ST 5', 'SIMS AVE', 'SIMS DR', 'SIMS PL', 'SIN MING AVE', 'SIN MING RD', 'SMITH ST', 'SPOTTISWOODE PK RD', "ST. GEORGE'S LANE", "ST. GEORGE'S RD", 'STIRLING RD', 'STRATHMORE AVE', 'SUMANG LANE', 'SUMANG LINK', 'SUMANG WALK', 'TAH CHING RD', 'TAMAN HO SWEE', 'TAMPINES AVE 1', 'TAMPINES AVE 4', 'TAMPINES AVE 5', 'TAMPINES AVE 7', 'TAMPINES AVE 8', 'TAMPINES AVE 9', 'TAMPINES CTRL 1', 'TAMPINES CTRL 7', 'TAMPINES CTRL 8', 'TAMPINES ST 11', 'TAMPINES ST 12', 'TAMPINES ST 21', 'TAMPINES ST 22', 'TAMPINES ST 23', 'TAMPINES ST 24', 'TAMPINES ST 32', 'TAMPINES ST 33', 'TAMPINES ST 34', 'TAMPINES ST 41', 'TAMPINES ST 42', 'TAMPINES ST 43', 'TAMPINES ST 44', 'TAMPINES ST 45', 'TAMPINES ST 61', 'TAMPINES ST 71', 'TAMPINES ST 72', 'TAMPINES ST 81', 'TAMPINES ST 82', 'TAMPINES ST 83', 'TAMPINES ST 84', 'TAMPINES ST 86', 'TAMPINES ST 91', 'TANGLIN HALT RD', 'TAO CHING RD', 'TEBAN GDNS RD', 'TECK WHYE AVE', 'TECK WHYE CRES', 'TECK WHYE LANE', 'TELOK BLANGAH CRES', 'TELOK BLANGAH DR', 'TELOK BLANGAH HTS', 'TELOK BLANGAH RISE', 'TELOK BLANGAH ST 31', 'TELOK BLANGAH WAY', 'TESSENSOHN RD', 'TG PAGAR PLAZA', 'TIONG BAHRU RD', 'TOA PAYOH CTRL', 'TOA PAYOH EAST', 'TOA PAYOH NTH', 'TOH GUAN RD', 'TOH YI DR', 'TOWNER RD', 'UBI AVE 1', 'UPP ALJUNIED LANE', 'UPP BOON KENG RD', 'UPP CROSS ST', 'UPP SERANGOON CRES', 'UPP SERANGOON RD', 'UPP SERANGOON VIEW', 'VEERASAMY RD', 'WATERLOO ST', 'WELLINGTON CIRCLE', 'WEST COAST DR',
                                                    'WEST COAST RD', 'WHAMPOA DR', 'WHAMPOA RD', 'WHAMPOA STH', 'WHAMPOA WEST', 'WOODLANDS AVE 1', 'WOODLANDS AVE 3', 'WOODLANDS AVE 4', 'WOODLANDS AVE 5', 'WOODLANDS AVE 6', 'WOODLANDS AVE 9', 'WOODLANDS CIRCLE', 'WOODLANDS CRES', 'WOODLANDS CTR RD', 'WOODLANDS DR 14', 'WOODLANDS DR 16', 'WOODLANDS DR 40', 'WOODLANDS DR 42', 'WOODLANDS DR 44', 'WOODLANDS DR 50', 'WOODLANDS DR 52', 'WOODLANDS DR 53', 'WOODLANDS DR 60', 'WOODLANDS DR 62', 'WOODLANDS DR 70', 'WOODLANDS DR 71', 'WOODLANDS DR 72', 'WOODLANDS DR 73', 'WOODLANDS DR 75', 'WOODLANDS RING RD', 'WOODLANDS RISE', 'WOODLANDS ST 11', 'WOODLANDS ST 13', 'WOODLANDS ST 31', 'WOODLANDS ST 32', 'WOODLANDS ST 41', 'WOODLANDS ST 81', 'WOODLANDS ST 82', 'WOODLANDS ST 83', 'YISHUN AVE 1', 'YISHUN AVE 11', 'YISHUN AVE 2', 'YISHUN AVE 3', 'YISHUN AVE 4', 'YISHUN AVE 5', 'YISHUN AVE 6', 'YISHUN AVE 7', 'YISHUN AVE 9', 'YISHUN CTRL', 'YISHUN CTRL 1', 'YISHUN RING RD', 'YISHUN ST 11', 'YISHUN ST 20', 'YISHUN ST 21', 'YISHUN ST 22', 'YISHUN ST 31', 'YISHUN ST 41', 'YISHUN ST 43', 'YISHUN ST 51', 'YISHUN ST 61', 'YISHUN ST 71', 'YISHUN ST 72', 'YISHUN ST 81', 'YUAN CHING RD', 'YUNG AN RD', 'YUNG HO RD', 'YUNG KUANG RD', 'YUNG LOH RD', 'YUNG PING RD', 'YUNG SHENG RD', 'ZION RD'])


    button= st.button("Predict the Price", use_container_width= True)

    if button:

            
        predicted_price= predict_resale_price(town,flat_type,block,street_name,floor_area_sqm,
                                        flat_model, lease_commence_date,storey_range_start,
                                        storey_range_end, resale_year,resale_month)

        st.write("## :green[**The Predicted Price is :**]",predicted_price)


elif select == "About":

    st.header(":blue[Data Collection and Preprocessing:]")
    st.write("Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for the years 1990 to Till Date. Preprocess the data to clean and structure it for machine learning.")

    st.header(":blue[Feature Engineering:]")
    st.write("Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date. Create any additional features that may enhance prediction accuracy.")
    
    st.header(":blue[Model Selection and Training:]")
    st.write("Choose an appropriate machine learning model for regression (e.g., linear regression, decision trees, or random forests). Train the model on the historical data, using a portion of the dataset for training.")

    st.header(":blue[Model Evaluation:]")
    st.write("Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score.")

    st.header(":blue[Streamlit Web Application:]")
    st.write("Develop a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.). Utilize the trained machine learning model to predict the resale price based on user inputs.")

    st.header(":blue[Deployment on Render:]")
    st.write("Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.")
    
    st.header(":blue[Testing and Validation:]")
    st.write("Thoroughly test the deployed application to ensure it functions correctly and provides accurate predictions.")


