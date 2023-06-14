import streamlit as st
import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import os
import sys
from PIL import Image
from matplotlib import pyplot as plt
import time
import datetime
from io import BytesIO
import uuid
from pyforest import *
import recordlinkage as rl
from recordlinkage.preprocessing import clean
from recordlinkage.index import SortedNeighbourhood
# import recordlinkage as rl
# from recordlinkage.preprocessing import clean
from tqdm import tqdm as tdm
#import spacy
from kneed import KneeLocator, DataGenerator as dg
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
from Back_end.back_end_functions import precleaning, clean_zip, cleaning_cols, Sorted_Neighbourhood_Prediction, elbow_function, merge_dataframes
from Front_end.front_end_functions import df_check_data, comp_time_eq, threshold_match_tbl, df_store_match, venn_diagram, mini_pie_charts, violin_graph, na_graphs

# 1- Main Window -- Layout Settings------------------------------------------------------------
st.set_page_config(layout="wide")
base="dark"
primaryColor="#BF2A7C" #PINK
backgroundColor="#FFFFFF" #MAIN WINDOW BACKGROUND COLOR (white)
secondaryBackgroundColor="#EBF3FC" #SIDEBAR COLOR (light blue)
textColor="#31333F"
secondaryColor="#F0F2F6" #dark_blue
tertiaryColor ="#0810A6"
light_pink = "#CDC9FA"
plot_blue_colour="#0810A6" #vibrant blue for plots

footer="""<style>.footer {
position: fixed;left: 0;bottom: 0;width: 100%;background-color: white;color: black;text-align: center;
}
</style>
<div class="footer">
<p>(c) 2023 Zeta Global, Dev Version 1.3, GDSA</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

#List of possible tresholds used in the algorithm
chosen_tresholds = ('0.5', '0.6', '0.75','0.85','0.95','0.97','0.99')


# -----------------------------------------------------------------------------------------------
# 2- Sidebar -- Parameter Settings---------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# Input Data 
st.sidebar.title("Input Data")



## Unmatched file: this are unknown customers 
unmatched_file = st.sidebar.file_uploader('Unmatched Dataset', type='csv', help='Dataset without email address')

unmatched_file_valid_flag = False
if unmatched_file is not None:
    # Check MIME type of the uploaded file
    if  unmatched_file.name == "unmatched_data.csv":
        unmatched_df = pd.read_csv(unmatched_file)
        unmatched_file_valid_flag = True
        st.sidebar.success('Success')
    else:
        st.sidebar.error('Error: please, re-upload file called unmatched_data.csv')

## Customer file: this are known customers (a.k.a customer databse with PII)
customer_file = st.sidebar.file_uploader("Customer Dataset", type='csv', help='Dataset containing email address')

customer_file_valid_flag = False
if customer_file is not None:
    # Check MIME type of the uploaded file
    if  customer_file.name == "customer_data.csv":
        customer_df = pd.read_csv(customer_file)
        customer_file_valid_flag = True
        st.sidebar.success('Success')
    else:
        st.sidebar.error('Error: please, re-upload file called customer_data.csv')



## Check Data Formats button
check_data = st.sidebar.button("""Check Data""", help = 'Show statistics of your data')


# Persist  unm/cust data statistics
if 'valid_flag' not in st.session_state:
    st.session_state.valid_flag = False

if 'unmatched_obj_cols' not in st.session_state:
    st.session_state.unmatched_obj_cols = ['']
if 'unmatched_num_cols' not in st.session_state:
    st.session_state.unmatched_num_cols = ['']
if 'unmatched_obj_cols_optional' not in st.session_state:
    st.session_state.unmatched_obj_cols_optional = ['']

if 'unmatched_df_com_cols' not in st.session_state:
    st.session_state.unmatched_df_com_cols = pd.DataFrame()
if 'customer_df_com_cols' not in st.session_state:
    st.session_state.customer_df_com_cols = pd.DataFrame()

if check_data:
    if (unmatched_file_valid_flag == True) and (customer_file_valid_flag ==True):

        unmatched_df, customer_df, unmatched_obj_cols, unmatched_num_cols, unmatched_obj_cols_optional, commun_cols_u_c= df_check_data(unmatched_df, customer_df)
    
        st.session_state.valid_flag = True
        st.session_state.unmatched_obj_cols = unmatched_obj_cols
        st.session_state.unmatched_num_cols = unmatched_num_cols
        st.session_state.unmatched_obj_cols_optional = unmatched_obj_cols_optional
        st.session_state.unmatched_df_com_cols = unmatched_df[commun_cols_u_c]
        st.session_state.customer_df_com_cols = customer_df[commun_cols_u_c]

    else:
        pass
else:
    pass



# Parameter Selection 
st.sidebar.title("Parameter Selection")
st.sidebar.subheader('Main field to be matched')

# sidebar columns 5 and 6
col5_sidebar, col6_sidebar= st.sidebar.columns([2, 2])

## Unmatched and Customer columns
if (unmatched_file_valid_flag == True) and (customer_file_valid_flag ==True):
    # Mail field to be matched
    select_box_unmatched_load_main = st.sidebar.selectbox(
        'Select column',
        options=st.session_state.unmatched_obj_cols, 
        help = 'Select main column to apply Match')
    
    ## Main Threshold
    selectbox_threshold = st.sidebar.selectbox('Threshold',
        chosen_tresholds, help='Select main field threshold to perfom match')

    # #Fixed threshold columns 
    st.sidebar.subheader('Run algorithm on')
    select_box_unmatched_load_11 = st.sidebar.selectbox(
        'Select extra column',
        (st.session_state.unmatched_num_cols), 
        help = 'Run algorithm on extra column for 100% threshold if desired'
        , key = 11)

    select_box_unmatched_load_12 = st.sidebar.selectbox(
        'Select extra column',
        (st.session_state.unmatched_num_cols), 
        help = 'Run algorithm on extra column for 100% threshold if desired'
        , key = 12)

    

    x = comp_time_eq(st.session_state['unmatched_df_com_cols'].shape[0], selectbox_threshold)
    if x > 0:
        st.sidebar.markdown(f'<h1 style="color:{tertiaryColor};font-size:16px;">{f"The operation will take {datetime.timedelta(seconds=x)}"}</h1>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f'<h1 style="color:{tertiaryColor};font-size:16px;">{f"The operation will take less than a minute"}</h1>', unsafe_allow_html=True)
    



elif unmatched_file_valid_flag == True:
    selectbox_mss = list([" "])

    # Mail field to be matched
    select_box_unmatched_load_main = st.sidebar.selectbox(
        'Select column',
        options=selectbox_mss, 
        help = 'Upload Customer Database and select columns to apply Match')
    
    selectbox_threshold = st.sidebar.selectbox('Threshold',
        chosen_tresholds, help='Select main field threshold to perfom match')

     # Run algorithm on  
    st.sidebar.subheader('Run algorithm on')
    select_box_unmatched_load_21 = st.sidebar.selectbox(
        'Select column',
        (selectbox_mss), 
        help = 'Upload Customer Database and select columns to apply Match'
        , key = 21)
    select_box_unmatched_load_22 = st.sidebar.selectbox(
        'Select column',
        (selectbox_mss), 
        help = 'Upload Customer Database and select columns to apply Match'
        , key = 22)

elif customer_file_valid_flag == True:
    selectbox_mss = list([" "])
    # Mail field to be matched
    
    select_box_unmatched_load_main = st.sidebar.selectbox(
        'Select column',
        options=selectbox_mss, 
        help = 'Upload Unmatched Dataset and select columns to apply Match')
    
    selectbox_threshold = st.sidebar.selectbox('Threshold',
        chosen_tresholds, help='Select main field threshold to perfom match')

     ## Run algorithm on  
    st.sidebar.subheader('Run algorithm on')
    select_box_unmatched_load_31 = st.sidebar.selectbox(
        'Select column',
        (selectbox_mss), 
        help = 'Upload Unmatched Dataset and select columns to apply Match'
        , key = 31)
    select_box_unmatched_load_32 = st.sidebar.selectbox(
        'Select column',
        (selectbox_mss), 
        help = 'Upload Unmatched Dataset and select columns to apply Match'
        , key = 32)
   
   
else:
    selectbox_mss = list([" "])
    # Mail field to be matched
    select_box_unmatched_main_empty = st.sidebar.selectbox(
        'Select column',
        options=selectbox_mss, help = 'Please, upload data to start')
    ## Main Threshold
    selectbox_threshold = st.sidebar.selectbox('Threshold',
        chosen_tresholds, help='Select main field threshold to perfom match')
    # Run algorithm on
    st.sidebar.subheader('Run algorithm on')
    select_box_unmatched_empty1 =st.sidebar.selectbox(
        """Select extra column"""
        , options=selectbox_mss , help='Please, upload data to start', key = 11)
    select_box_customer_empty1 = st.sidebar.selectbox(
        'Select extra column'
        , options=selectbox_mss , help='Please, upload data to start', key = 12)
    st.sidebar.write(' ')

## Apply Match button
st.sidebar.title("Match data")
match_button = st.sidebar.button("""Apply Match""", help='Apply match to your data and show the results')

st.sidebar.text(" ")


# -----------------------------------------------------------------------------------------------
# 1- Main Window -- Parameter Settings-----------------------------------------------------------
# ----------------------------------------------------------------------------------------------- 
# st.title(f'My first app {st.__version__}')


# Creating columns 1 and 2
col1, col2 = st.columns([13, 2])

## Zeta Logo
#zeta_logo = Image.open('ZETA_BIG-99e027c9.webp') #white logo 
zeta_logo = Image.open('ZETA_BIG-99e027c92.png') #blue logo 
col2.image(zeta_logo)

## Header
col1.title("Zeta Customer Matcher")
"""This app demonstrates Customers Probabilistic Matching Project"""

# Creating columns 3 and 4
col_space1, col_space2  =  st.columns([2,2])

# Creating columns 3 and 4
col3, col4= st.columns([2, 2])
colna1, colna2 = st.columns([2,2]) # NA message if no NAs

# Create col expand data
col_left_expander, col_right_expander = st.columns(2)

# Creating columns 5 and 6

col_threshold_left, col_threshold_rigt = st.columns([2,2])

# Creating columns 5 and 6

col5, col6 = st.columns([2,2])


## Summary pie charts
colp1, colp2, colp3 = st.columns([2, 2, 2])


# Set num of column algorithm will compare, by default is 1
col_compare_alg = 1


# Display features if data is valid and match button is clicked
if match_button and unmatched_file_valid_flag and customer_file_valid_flag:
    if (select_box_unmatched_load_11 == select_box_unmatched_load_12 )and ((select_box_unmatched_load_11 != '') or select_box_unmatched_load_12):
        st.error('Please, select two different extra columns to perform match')
    else:
        if select_box_unmatched_load_main and (select_box_unmatched_load_11 or select_box_unmatched_load_12):
            col_compare_alg = 2 
        if select_box_unmatched_load_main and select_box_unmatched_load_11 and select_box_unmatched_load_12:
            col_compare_alg = 3
        
        # Bar progress
        latest_iteration = st.empty()
        bar = st.progress(0)
        latest_iteration.text('Matching in progress...')


        
        
        #------ THIS IS PRE-DATA CLEANING
        
        #ALREADY TRANSFORMED IN BACK END FUNCTIONS!!!
        unmatched_df = precleaning(unmatched_df)
        customer_df = precleaning(customer_df)
        
        
        #CLEAN ZIP codes
        for col in unmatched_df.columns:
            if 'STORE_ZIP' in col.lower():
                clean_zip(unmatched_df, col='STORE_ZIP')
        
        #clean zip codes        
        for col in unmatched_df.columns:
            if 'STORE_ZIP' in col.lower():
                clean_zip(unmatched_df, col='STORE_ZIP')
                
         #Cleaning bad names
        values_to_remove = ["DISCOVER CARDMEMBER", "CHASE VISA CARDHOLDER", "VALUE LINK CTS", "VISA CARDHOLDER", "VALUED CUSTOMER"]
        # Create a boolean mask indicating rows to keep
        mask_un = ~unmatched_df['MSRNAME'].isin(values_to_remove)
        # Apply the mask to the DataFrame
        unmatched_df = unmatched_df[mask_un]       
        # Create a boolean mask indicating rows to keep
        mask_cu = ~customer_df['MSRNAME'].isin(values_to_remove)
        # Apply the mask to the DataFrame
        customer_df = customer_df[mask_cu]       


        ## REMOVE NaNs FROM SELECTED COLS TO JOIN ON
        try: # try to avoid error if select box left blank
            unmatched_df = unmatched_df[unmatched_df[select_box_unmatched_load_main].notnull()]
        except:
            pass
        try:
            unmatched_df = unmatched_df[unmatched_df[select_box_unmatched_load_11].notnull()]
        except:
            pass
        try:
            unmatched_df = unmatched_df[unmatched_df[select_box_unmatched_load_12].notnull()]
        except:
            pass
          
        

            

    ## <- Data Cleaning end
        #latest_iteration = st.empty()
        bar.progress((100//4)*1)
        latest_iteration.text('Initiating algorithm...')

        
        #FIRST RUN OF THE ALGORITHM FOR THE CHOSEN THRESHOLD
        model_thld = float(selectbox_threshold)
        vectors,predictions= Sorted_Neighbourhood_Prediction(unmatched_df,
                                                            customer_df,
                                                            pred_comp =col_compare_alg,
                                                            threshold=model_thld,
                                                            method_str="jarowinkler",
                                                            method_num="step",
                                                            scale=5,
                                                            offset=5,
                                                            main_field_compare = select_box_unmatched_load_main,
                                                            select_box_unmatched_load_11=select_box_unmatched_load_11,
                                                            select_box_unmatched_load_12=select_box_unmatched_load_12)
        
        
        #latest_iteration = st.empty()
        bar.progress((100//4)*3)
        latest_iteration.text('Calculating number of matches...')

        #merge matching
        df_final = merge_dataframes(predictions, unmatched_df, customer_df)

        # SECOND RUN OF THE ALGORITHM FOR ALL THE THRESHOLD
        # This will help us building elbow plot and the table below
        threshold_possibilities = list(chosen_tresholds)
        count_num_mtch = []


        for possibility in threshold_possibilities:             
            vectors,predictions= Sorted_Neighbourhood_Prediction(unmatched_df,
                                                                customer_df,
                                                                pred_comp =col_compare_alg,
                                                                threshold=possibility,
                                                                method_str="jarowinkler",
                                                                method_num="step",
                                                                scale=5,
                                                                offset=5,
                                                                main_field_compare = select_box_unmatched_load_main,
                                                                select_box_unmatched_load_11=select_box_unmatched_load_11,
                                                                select_box_unmatched_load_12=select_box_unmatched_load_12)

            #merge matching
            df_final_all_trhld = merge_dataframes(predictions, unmatched_df, customer_df)
            # count number of matches per threshold 
            count_num_mtch.append(len(df_final_all_trhld))
     
            
        
        # Create threshold vs #matches table
        match_dict, best_threshold_df, best_threshold_df_T = threshold_match_tbl(threshold_possibilities,count_num_mtch )


        #Best threshold and Elbow chart   
        best_threshold, elbow_chart= elbow_function(best_threshold_df_T,
                                                    'Threshold', 
                                                    '#Match', 
                                                    backgroundColor, 
                                                    plot_blue_colour, 
                                                    primaryColor, 
                                                    textColor)

        

        ## Example Fuzzy Logic output table 
        algrtm_output_df = df_final.copy()
        
        if select_box_unmatched_load_main and select_box_unmatched_load_11 and select_box_unmatched_load_12:     
            algrtm_output_df = algrtm_output_df[[f'{select_box_unmatched_load_main}_MTCH', 
                                                    f'{select_box_unmatched_load_main}_UNMTCH', 
                                                    f'{select_box_unmatched_load_11}_MTCH', 
                                                    f'{select_box_unmatched_load_11}_UNMTCH',
                                                    f'{select_box_unmatched_load_12}_MTCH',  
                                                    f'{select_box_unmatched_load_12}_UNMTCH']]

        elif select_box_unmatched_load_main and select_box_unmatched_load_11:
            algrtm_output_df = algrtm_output_df[[f'{select_box_unmatched_load_main}_MTCH', 
                                                    f'{select_box_unmatched_load_main}_UNMTCH', 
                                                    f'{select_box_unmatched_load_11}_MTCH', 
                                                    f'{select_box_unmatched_load_11}_UNMTCH']]

        elif select_box_unmatched_load_main and select_box_unmatched_load_12:
            algrtm_output_df = algrtm_output_df[[f'{select_box_unmatched_load_main}_MTCH', 
                                                    f'{select_box_unmatched_load_main}_UNMTCH', 
                                                    f'{select_box_unmatched_load_12}_MTCH', 
                                                    f'{select_box_unmatched_load_12}_UNMTCH']]

        elif select_box_unmatched_load_main:
            algrtm_output_df = algrtm_output_df[[f'{select_box_unmatched_load_main}_MTCH', 
                                                    f'{select_box_unmatched_load_main}_UNMTCH']]

        
        df_copy, df_gb_store_mtch= df_store_match (df_final,select_box_unmatched_load_main)
        


        # Col 3 text
        col_threshold_left.subheader(f'Perform Matching at Threshold: {selectbox_threshold}')
        col_threshold_left.markdown(f'Optimal Threshold: {best_threshold} (based on number of matches vs quality)')
        col5.subheader(' ')

        #latest_iteration = st.empty()
        bar.progress((100//4)*4)
        latest_iteration.text('Process completed!')

        # Display Elbow chart
        # elbow_plot_title = '<p style="font-family:sans-serif;color:black; font-size: 20px; text-align: center;">Number Matched Records per Threshold</p>'
        # col5.markdown(elbow_plot_title, unsafe_allow_html=True)
        col5.pyplot(elbow_chart)

        #Display best_threshold_df table
        #styler = best_threshold_df.style.hide_index()
        #col5.write(styler.to_html(), unsafe_allow_html=True) 
        hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        # Display a static table
        col5.table(best_threshold_df)
        
        # /....
        # Calculations to display Match Venn Graph
        match_count = np.round(match_dict[selectbox_threshold], 1)
        unmatched_df_count = np.round((len(st.session_state['unmatched_df_com_cols']) - match_count), 1)
        matched_df_count = np.round((len(st.session_state.customer_df_com_cols) - match_count), 1)

        fig_venn = venn_diagram(unmatched_df_count,matched_df_count,match_count)
        
        
        # Display Match Venn Graph ONLY if unmatched and customer .csv are uploaded and the threshold was set
        col6.subheader(' ')
        col6.pyplot(fig_venn)
        # /....

        bar.empty()
        latest_iteration.empty()

        # Display dataset summary after Probabilistic Matching
        st.subheader('Dataset Matched Summary​')


        # /....
        #Autocorrect Statistics ----------------------------------------
        unmtch_name_col = st.session_state.unmatched_df_com_cols.MSRNAME
        ctomer_name_col = st.session_state.customer_df_com_cols.MSRNAME

        ## Summary pie charts
        colp1, colp2, colp3 = st.columns([2, 2, 2])

        buf1, buf2, buf3 = mini_pie_charts(df_final, select_box_unmatched_load_main, unmtch_name_col, ctomer_name_col )

        #y = np.array([unmtch_name_col.nunique(), len(unmtch_name_col) - unmtch_name_col.nunique()])
        #customer_matches = pd.DataFrame({"labels" : ['Only one match','More than one match'],"values":[len(df_final[f'{select_box_unmatched_load_main}_MTCH'].unique()), len(df_final)- len(df_final[f'{select_box_unmatched_load_main}_MTCH'].unique()) ]})
        y1= np.array([len(df_final[f'{select_box_unmatched_load_main}_MTCH'].unique()), len(df_final)- len(df_final[f'{select_box_unmatched_load_main}_MTCH'].unique()) ]) 
        labels1= ['Only one match','More than one match']

           #len(df_final[f'{select_box_unmatched_load_main}_MTCH'].unique()) -> Only one match
           #len(df_final)- len(df_final[f'{select_box_unmatched_load_main}_MTCH'].unique()) -> more than one match
        mycolors = [plot_blue_colour, primaryColor]
        explode = (0, 0)
    
        #fig1, ax1 = plt.subplots()
        fig1, ax1 = plt.subplots(figsize=(6,6))
        fig1.patch.set_facecolor(backgroundColor)
    
        ax1.pie(y1, autopct=lambda x: '{:.0f}'.format(x*y1.sum()/100), explode=explode, labels=None,
            shadow=False, startangle=90, colors=mycolors, textprops={'color':"w",'fontsize': 14} ) 
        ax1.axis('equal')
        plt.title('MATCH', color=textColor, y=1.1, fontsize=16)
        plt.legend(labels1, loc='lower left', bbox_to_anchor=(0.5, -0.08))
        #plt.figure(figsize=(10,10))

        buf1 = BytesIO()
        fig1.savefig(buf1, format="png")

        #y = np.array([unmtch_name_col.nunique(), len(unmtch_name_col) - unmtch_name_col.nunique()])
        #unmatched_matches = pd.DataFrame({"labels" : ['Only one match','More than one match'],"values":[len(df_final[f'{select_box_unmatched_load_main}_UNMTCH'].unique()), len(df_final)- len(df_final[f'{select_box_unmatched_load_main}_UNMTCH'].unique())]})
        y2=np.array([len(df_final[f'{select_box_unmatched_load_main}_UNMTCH'].unique()), len(df_final)- len(df_final[f'{select_box_unmatched_load_main}_UNMTCH'].unique())])
        labels2=['Only one match','More than one match']

           #len(df_final[f'{select_box_unmatched_load_main}_UNMTCH'].unique()) -> Only one match
           #len(df_final)- len(df_final[f'{select_box_unmatched_load_main}_UNMTCH'].unique()) -> more than one match
        mycolors = [plot_blue_colour, primaryColor]
        explode = (0, 0)
    
        #fig2, ax2 = plt.subplots()
        fig2, ax2 = plt.subplots(figsize=(6,6))
        fig2.patch.set_facecolor(backgroundColor)
    
        ax2.pie(y2, autopct=lambda x: '{:.0f}'.format(x*y2.sum()/100), explode=explode, labels=None,
            shadow=False, startangle=90, colors=mycolors, textprops={'color':"w",'fontsize': 14}, radius=1800)
        ax2.axis('equal')
        plt.title('UNMATCH', color=textColor, y=1.1, fontsize=16)
        plt.legend(labels2, loc='lower left', bbox_to_anchor=(0.5, -0.08))
        #plt.figure(figsize=(10,10))
    
        buf2 = BytesIO()
        fig2.savefig(buf2, format="png")

        colp1.image(buf1)
        colp2.image(buf2)
        colp3.image(buf3)

        with colp1:
            st.write('')
        with colp2:
            st.write('')
        with colp3:
            st.write('')

        st.dataframe(data=algrtm_output_df.head(100), width=None, height=None)

        st.subheader('Store Performance​')

        #----- Col 7 and Col 8--------
        col7, col8 = st.columns([2, 2])
        store_table_title = '<p style="font-family:sans-serif;color:#30343F; font-size: 19px; text-align: left;">Store Statistics Table</p>'
        col7.markdown(store_table_title, unsafe_allow_html=True)
        violin_chart_title = '<p style="font-family:sans-serif;color:#30343F; font-size: 19px; text-align: left;">Store Statistics Distribution</p>'
        col8.markdown(violin_chart_title, unsafe_allow_html=True)
        #col8.write("Store Statistics Distribution")
        col7.dataframe(data=df_gb_store_mtch)

        # /....
        # Violin plot
        ax_violin_plot= violin_graph(df_gb_store_mtch)

        col8.pyplot(ax_violin_plot)


        
     # /....
    
elif match_button and (unmatched_file_valid_flag or customer_file_valid_flag):
    col_space1.error('Please, enter valid data and click "Apply Match" to perform matching')
elif unmatched_file_valid_flag and customer_file_valid_flag:
    col_space1.info('Please, check data and apply match')
    st.empty()
else:
    col_space1.info('Please, upload Unmatched Dataset and Customer Dataset to start')
    col_space1.empty()


if st.session_state['valid_flag']:
    #If both files are uploaded print stats of each one
    if  unmatched_file_valid_flag and customer_file_valid_flag:
        col_space1.success('Format Check Completed')    
        df_info = {'Number of rows':[len(st.session_state['unmatched_df_com_cols']), len(st.session_state.customer_df_com_cols)],
                    'Number of columns': [len(st.session_state['unmatched_df_com_cols'].columns), len(st.session_state.customer_df_com_cols.columns)]}
        df_info = pd.DataFrame(df_info).transpose().rename(columns={0:'Unmatch Data', 1:'Customer Data'})
        #col_space1.dataframe(df_info)

        ## NA charts
        null_df_unmatched = st.session_state['unmatched_df_com_cols'].apply(lambda x: sum(x.isnull())).to_frame(name='count_null').reset_index()
        null_df_customers = st.session_state.customer_df_com_cols.apply(lambda x: sum(x.isnull())).to_frame(name='count_null').reset_index()

        #null_counts = df.isnull().sum()
        notnull_df_unmatched = st.session_state['unmatched_df_com_cols'].apply(lambda x: sum(x.notnull())).to_frame(name='count').reset_index()
        notnull_df_customers = st.session_state.customer_df_com_cols.apply(lambda x: sum(x.notnull())).to_frame(name='count').reset_index()

        # create new DataFrame with counts
        #counts_df = pd.DataFrame({'null_counts': null_counts,
        #                  'notnull_counts': notnull_counts})

        merged_unmatched = pd.merge(null_df_unmatched, notnull_df_unmatched, on='index')
        merged_customer = pd.merge(null_df_customers, notnull_df_customers, on='index')
        #merged_unmatched1=merged_unmatched.melt('index',var_name='type',value_name='count')
        #merged_customer1=merged_customer.melt('index',var_name='type',value_name='count')

        unmatched_city_count = st.session_state.unmatched_df_com_cols.groupby(['STORE_CITY'])['STORE_CITY'].count().sort_values(ascending=False).reset_index(name='counts').head(10)
        customer_city_count = st.session_state.customer_df_com_cols.groupby(['STORE_CITY'])['STORE_CITY'].count().sort_values(ascending=False).reset_index(name='counts')#.head(10)
        # Merge DataFrame A and DataFrame B on 'city_store' column
        merged_df = pd.merge(unmatched_city_count, customer_city_count, on='STORE_CITY')
        # Create final DataFrame B with selected columns
        final_customer_city_count = merged_df[['STORE_CITY', 'counts_y']].rename(columns={'counts_y': 'counts'})
        
        merged_df_un = pd.merge(st.session_state.unmatched_df_com_cols, unmatched_city_count, on='STORE_CITY', how='inner')
        merged_df_cu = pd.merge(st.session_state.customer_df_com_cols, final_customer_city_count, on='STORE_CITY', how='inner')
        
        ctomer_name_col = st.session_state.customer_df_com_cols.MSRNAME
        customer_unique = pd.DataFrame({"labels" : ['Known unique customers', 'Duplicates'],"values":[ctomer_name_col.nunique(), len(ctomer_name_col) - ctomer_name_col.nunique()]})

        unmtch_name_col = st.session_state.unmatched_df_com_cols.MSRNAME
        unmatched_unique = pd.DataFrame({"labels" : ['Unmatched unique customers', 'Duplicates'],"values":[unmtch_name_col.nunique(), len(unmtch_name_col) - unmtch_name_col.nunique()]})

        #merged_unmatched['count'].idxmax()
        #merged_customer['count'].idxmax()
        cmap = {
        'Unmatched unique customers':  '#0810A6',
        'Duplicates': '#BF2A7C',
    }
        
        #na_unmatched=alt.Chart(merged_unmatched1).mark_bar(size=20).encode(
        #    x=alt.X('count:Q', scale= alt.Scale(domainMax= len(st.session_state['unmatched_df_com_cols']) )),
        #    y='type:N',  
        #    color=alt.Color('type:N', scale=alt.Scale(domain=list(cmap.keys()), range=list(cmap.values()))) ,
        #    tooltip=['count'] ,
        #    row=alt.Row('index:N', title=None)
        #).properties(width=500 , height=alt.Step(25) ).configure_axis(title=None, grid=False)

        #Radial chart with Store_city and its count
        base = alt.Chart(unmatched_city_count, title="Top 10 store cities by count").encode(
        theta=alt.Theta("counts:Q", stack=True ),
        radius=alt.Radius("counts", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
        color=alt.Color ("STORE_CITY:N" ,title="Cities"),
        order= alt.Order("counts:Q", sort="ascending")
        )
        c1 = base.mark_arc(innerRadius=20, stroke="#fff")
        c2 = base.mark_text(radiusOffset=10).encode(text="counts:Q")

        #Stack bar with city, Numshop
        bars = alt.Chart(merged_df_un.dropna(), title="Distribution of num_shop by top 10 cities").mark_bar().encode(
        x=alt.X('count()', stack='zero',title=None),
        y=alt.Y('STORE_CITY:N',title=None),
        color=alt.Color('NUM_SHOP:N', bin=alt.Bin(maxbins=5), scale=alt.Scale(scheme='plasma'))
            )
        
        #Mini pie chart 1
        base1 = alt.Chart(unmatched_unique).encode(
        theta= alt.Theta("values:Q",stack=True),
        color = alt.Color('labels:N' ,title=None, scale=alt.Scale(domain=list(cmap.keys()), range=list(cmap.values()))) ,
        )

        pie = base1.mark_arc(outerRadius=120)
        text = base1.mark_text(radius=140, size=20).encode(text="values:N")

        

        #na_unmatched = na_graphs(null_df_unmatched, name='unmatched')
        with col3:
            #st.altair_chart(na_unmatched, use_container_width=False)
            #null_df_unmatched
            #notnull_df_unmatched
            #merged_unmatched
            #merged_unmatched1
            st.subheader('Unmatched dataset')
            st.altair_chart((c1+c2), use_container_width=True)
            st.altair_chart((bars), use_container_width=True)
            st.altair_chart((pie+text), use_container_width=True)
            #pie chart
            st.write('')

        #na_customer = na_graphs(null_df_customers, name='customer')
        #na_customer=alt.Chart(merged_customer1).mark_bar(size=20).encode(
        #    x=alt.X('count:Q', scale= alt.Scale(domainMax= len(st.session_state.customer_df_com_cols) )),
        #    y='type:N',  
        #    color=alt.Color('type:N', scale=alt.Scale(domain=list(cmap.keys()), range=list(cmap.values()))) ,
        #    tooltip=['count'] ,
        #    row=alt.Row('index:N', title=None)
        #).properties(width=500, height=alt.Step(25)).configure_axis(title=None, grid=False)

        #Radial chart with Store_city and its count
        base_c = alt.Chart(final_customer_city_count, title='Store cities by count , based on unmatched data set').encode(
        theta=alt.Theta("counts:Q", stack=True ),
        radius=alt.Radius("counts", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
        color=alt.Color ("STORE_CITY:N" ,title="Cities"),
        order= alt.Order("counts:Q", sort="ascending")
        )
        c1_c = base_c.mark_arc(innerRadius=20, stroke="#fff")
        c2_c = base_c.mark_text(radiusOffset=10).encode(text="counts:Q")

        cmap = {
        'Known unique customers': '#0810A6',
        'Duplicates':  '#BF2A7C',
    }
        #Stack bar with city, Numshop
        bars_2 = alt.Chart(merged_df_cu.dropna(), title="Distribution of num_shop by cities").mark_bar().encode(
        x=alt.X('count()', stack='zero',title=None),
        y=alt.Y('STORE_CITY:N',title=None),
        color=alt.Color('NUM_SHOP:N', bin=alt.Bin(maxbins=5), scale=alt.Scale(scheme='plasma'))
            )
        
        #Mini pie chart 2
        base_2 = alt.Chart(customer_unique).encode(
        theta= alt.Theta("values:Q", stack=True),
        color= alt.Color('labels:N',title=None,scale=alt.Scale(domain=list(cmap.keys()), range=list(cmap.values()))) ,
        )

        pie_2 = base_2.mark_arc(outerRadius=120)
        text_2 = base_2.mark_text(radius=140, size=20).encode(text="values:N")

    
        

        with col4:
            #st.altair_chart(na_customer, use_container_width=False)
            #null_df_customers
            #notnull_df_customers
            #merged_customer
            #merged_customer1
            st.subheader('Customer dataset')
            st.altair_chart((c1_c+c2_c), use_container_width=True)
            st.altair_chart((bars_2), use_container_width=True)
            st.altair_chart((pie_2+text_2), use_container_width=True)
            st.write('')
        
        merged_customer_df  = merged_customer.assign(df=np.full(len(merged_customer), 'Customer'))
        merged_unmatched_df  = merged_unmatched.assign(df=np.full(len(merged_unmatched), 'Unmatched'))

        result = pd.concat([merged_customer_df, merged_unmatched_df]).reset_index()
        
        

        #na_chart = alt.Chart(result).mark_bar().encode(
        #x='sum(count):Q',
        #y='df:N',
        #color=alt.Color('type', scale=alt.Scale(domain=list(cmap.keys()), range=list(cmap.values()))) ,
        #row='index:N'
        #).configure_axis(title=None, grid=False).properties(width=700, height=70 )

        #st.altair_chart(na_chart)
        #st.write('')

        # Unmatch df
        col_left_expander.write('Unmatched data')
        with col_left_expander.expander("Expand data and statistics"):
            #Display unmatch df
            unmatched_df['STORE'] = unmatched_df['STORE'].round(0).astype(str).str[:-2]
            unmatched_df['STORE'] = unmatched_df['STORE'].replace('n', np.nan)
            unmatched_df['STORE_ZIP'] = unmatched_df['STORE_ZIP'].round(0).astype(str).str[:-2]
            unmatched_df['STORE_ZIP'] = unmatched_df['STORE_ZIP'].replace('n', np.nan)
            unmatched_df['NUM_SHOP'] = unmatched_df['NUM_SHOP'].round(0).astype(str).str[:-2]
            unmatched_df['NUM_SHOP'] = unmatched_df['NUM_SHOP'].replace('n', np.nan)
            st.dataframe(unmatched_df[commun_cols_u_c].head(100))
            #st.dataframe(st.session_state['unmatched_df_com_cols'].head(100))
            #Display unmatch df NAs 
            st.write(f'Number of Nan values in Unmatched Dataset: ')
            na_unmt_df = pd.DataFrame(st.session_state['unmatched_df_com_cols'].isna().sum())
            na_unmt_df = na_unmt_df.rename(columns={0:'#NAs'})
            st.dataframe(na_unmt_df)
        
        if len(na_unmt_df[na_unmt_df['#NAs']>0]) == 0:
            colna1.info('No NAs found in Unmatched Dataset')

        # Customer df
        col_right_expander.write(f'Customer data')
        with col_right_expander.expander("Expand data and statistics"):
            #Display customer df
            customer_df['STORE'] = customer_df['STORE'].round(0).astype(str).str[:-2]
            customer_df['STORE'] = customer_df['STORE'].replace('n', np.nan)
            customer_df['STORE_ZIP'] = customer_df['STORE_ZIP'].round(0).astype(str).str[:-2]
            customer_df['STORE_ZIP'] = customer_df['STORE_ZIP'].replace('n', np.nan)
            customer_df['NUM_SHOP'] = customer_df['NUM_SHOP'].round(0).astype(str).str[:-2]
            customer_df['NUM_SHOP'] = customer_df['NUM_SHOP'].replace('n', np.nan)
            st.dataframe(customer_df[commun_cols_u_c].head(100))
            #st.dataframe(st.session_state.customer_df_com_cols.head(100))
            #Display unmatch df NAs 
            st.write(f'Number of Nan values in Customer Data: ')
            na_cust_df = pd.DataFrame(st.session_state.customer_df_com_cols.isna().sum())
            na_cust_df = na_cust_df.rename(columns={0:'#NAs'})
            st.dataframe(na_cust_df)
        if len(na_cust_df[na_cust_df['#NAs']>0]) == 0:
            colna2.info('No NAs found in Customer Dataset')


    # Print unmatched_file stats if uploaded
    elif unmatched_file_valid_flag:
        col_space1.success('Format Check Completed')
        col_left_expander.write('Unmatched data')
        with col_left_expander.expander("Expand data and statistics"):
            #Display unmatch df
            unmatched_df['STORE'] = unmatched_df['STORE'].round(0).astype(str).str[:-2]
            unmatched_df['STORE'] = unmatched_df['STORE'].replace('n', np.nan)
            unmatched_df['STORE_ZIP'] = unmatched_df['STORE_ZIP'].round(0).astype(str).str[:-2]
            unmatched_df['STORE_ZIP'] = unmatched_df['STORE_ZIP'].replace('n', np.nan)
            unmatched_df['NUM_SHOP'] = unmatched_df['NUM_SHOP'].round(0).astype(str).str[:-2]
            unmatched_df['NUM_SHOP'] = unmatched_df['NUM_SHOP'].replace('n', np.nan)
            st.dataframe(unmatched_df[commun_cols_u_c].head(100))
            #st.dataframe(st.session_state['unmatched_df_com_cols'].head(100))
            #Display unmatch df NAs 
            st.write(f'Number of Nan values in Unmatched Dataset: ')
            na_unmt_df = pd.DataFrame(st.session_state['unmatched_df_com_cols'].isna().sum())
            na_unmt_df = na_unmt_df.rename(columns={0:'#NAs'})
            st.dataframe(na_unmt_df)

    # Print customer_file stats if uploaded
    elif customer_file_valid_flag:
        col_space1.success('Format Check Completed')
        col_right_expander.write(f'Customer data')
        with col_right_expander.expander("Expand data and statistics"):
            #Display customer df
            customer_df['STORE'] = customer_df['STORE'].round(0).astype(str).str[:-2]
            customer_df['STORE'] = customer_df['STORE'].replace('n', np.nan)
            customer_df['STORE_ZIP'] = customer_df['STORE_ZIP'].round(0).astype(str).str[:-2]
            customer_df['STORE_ZIP'] = customer_df['STORE_ZIP'].replace('n', np.nan)
            customer_df['NUM_SHOP'] = customer_df['NUM_SHOP'].round(0).astype(str).str[:-2]
            customer_df['NUM_SHOP'] = customer_df['NUM_SHOP'].replace('n', np.nan)
            st.dataframe(customer_df[commun_cols_u_c].head(100))
            #st.dataframe(st.session_state.customer_df_com_cols.head(100))
            #Display unmatch df NAs 
            st.write(f'Number of Nan values in Customer Data: ')
            na_cust_df = pd.DataFrame(st.session_state.customer_df_com_cols.isna().sum())
            na_cust_df = na_cust_df.rename(columns={0:'#NAs'})
            st.dataframe(na_cust_df)

    else: 
         pass
else:
    col3.write('')


