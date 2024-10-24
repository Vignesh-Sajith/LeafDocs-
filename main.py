import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
from PIL import Image
import gdown 

# Function to download the model from Google Drive
def download_model():
    # Replace 'YOUR_MODEL_FILE_ID' with the actual file ID from the shareable link
    url = 'https://drive.google.com/uc?id=1mffo2eJfe4XFUXxWra7zZtBSwqIe-U4r'
    output = 'trained_plant_disease_model.keras'
    gdown.download(url, output, quiet=False)

# Load Model & Scaler & Polynomial Features for Crop Yield Prediction
crop_yield_model = joblib.load('model.pkl')
sc = joblib.load('sc.pkl')
pf = joblib.load('pf.pkl')
df_final = pd.read_csv('test.csv')
df_main = pd.read_csv('main.csv')
yield_prediction_image = Image.open('img.png')
# Download and load the plant disease model
download_model()  # Call the download function
model = tf.keras.models.load_model("trained_plant_disease_model.keras")

# Tensorflow Model Prediction for Plant Disease
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Update columns for Crop Yield Prediction
def update_columns(df, true_columns):
    df[true_columns] = True
    other_columns = df.columns.difference(true_columns)
    df[other_columns] = False
    return df

# Prediction function for Crop Yield
def prediction(input):
    categorical_col = input[:2]
    input_df = pd.DataFrame({'average_rainfall': input[2], 'presticides_tonnes': input[3], 'avg_temp': input[4]}, index=[0])
    input_df1 = df_final.head(1)
    input_df1 = input_df1.iloc[:, 3:]
    true_columns = [f'Country_{categorical_col[0]}', f'Item_{categorical_col[1]}']
    input_df2 = update_columns(input_df1, true_columns)
    final_df = pd.concat([input_df, input_df2], axis=1)
    final_df = final_df.values
    test_input = sc.transform(final_df)
    test_input1 = pf.transform(test_input)
    predict = crop_yield_model.predict(test_input1)
    result = int(((predict[0] / 100) * 2.47105) * 100) / 100
    return (f"The Production of Crop Yields: {result} quintel/acres yield Production. "
            f"That means 1 acre of land produces {result} quintel of yield crop. It's all dependent on parameters like average rainfall, temperature, soil, etc.")

# Streamlit Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page",
    [
        "Home", "About", "Disease Recognition",
        "Crop Yield Prediction", "Pest Outbreak Prediction", 
        "Farmer Support Schemes"
    ]
)

# Plant Disease Project - Main Pages
if app_mode == "Home":
    st.header("PLANT LEAF DIESEASE DETECTION SYSTEM (PLDS)")
    image_path = "home_page.jpeg"
    st.image(image_path, width = 1000)
    st.markdown(""" Welcome to the Plant Leaf Disease Detection System (PLDDS) Otherwise formelly known as leafdoc🌿🔍
    
                 LeafDoc is a lightweight, AI-driven program designed to assist farmers and plant enthusiasts in identifying diseases affecting 
                 plants through their leaves. This project utilizes machine learning techniques to detect common diseases and assess the crop yield 
                 of various plants, enhancing crop management and overall plant health. Users can access the service via a web app, allowing them to 
                 scan images of their plant leaves for disease detection.



                 ### The Team (Sharjah Indian School Boys - Juwaiza )
                 1. Our little team , consisting of Vignesh, Ken and Prabhav are a few tinkering programmers who got inspired to take on some 
                    some big challenges in the area of machine learning, CNN (Convolution Neural Network), Image Classification etc. 
                    to solve some major problems in the largest employemnt sector of India, Agriculture. 
                 2. The devastating crop loss of multiple distasters in the field of agriculture; have motivated us to make this
                    as our final project for this exhibition, a remarkable tool that every farmer who toils hard will appreciate and use.
                 3. We intend this project to be a testimoney towards the intense need for integration of latest AI and ML technologies 
                    in the agriculture sectors, we as students are limited, more than a useful tool, our intention with this project is 
                    to spread an idea, that will reverberate throughout the economy.
                 4. We kindly request anyone who may have stumbled upon our little achievement, to share to for the benefit of others.

                 ### Why Choose Us?
                 - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
                 - **User-Friendly:** Simple and intuitive interface for seamless user experience.
                 - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

                 ### Our Services:
                 1. Our main service is leaf diesease detection program that we address as "LeafDoc", which can be accessed via the Diesease 
                  Recognition Page.
                 2. We Have also made our own independent datasets using Perplexity to make a Pest Outbreak Early Warning System.
                 3. By extracting Databases from kaggle, a complex crop yeild prediction software is also availible.
                 4. For the benefit of rural farmers, we have integrated a special tab, used for gaining intel on latest Governemnt Schemes. 

                 ### How did we make it?
                 You may wonder how we managed to pull such a project with no experience, I will provide a bibliography that contains links to
                 the various resources that was used by us throughout the duration of the project. But in short, a few stressfull weeks, 
                 many cups of coffee ☕ and lots of stackoverflow scrolling seemed to have done the trick :)
    
                 This project would never have been possible if it werent for the contunuing support of our teachers and senior coders who helped 
                 us make our vision come to life. I sincerlly Thank them all!""")

elif (app_mode == "About"):
    st.header("About")
    st.markdown("""
        #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo. 
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset 
                is divided into an 80/20 ratio of training and validation sets, preserving the directory structure. 
                A new directory containing 33 test images is created later for prediction purposes.
                
                #### Content
                1. **Train** (70,295 images)
                2. **Test** (33 images)
                3. **Validation** (17,572 images)

                #### HOW TO USE THE PLANT DISEASE DETECTION AI
                1. **First**, pluck the leaf and place it on a surface with a black or dark background, ensuring good lighting.
                2. **Take a picture** and upload the image on the website. Our AI model uses Convolutional Neural Networks (CNN) 
                    and image classification to identify the disease of the plant.
                3. **How to use the crop yield prediction AI**: Select your country and the crop to be grown from the dropdown list. 
                    Enter details such as annual rainfall (in mm per year), average temperature, and the amount of pesticides used per 
                    tonne (tonnes of active ingredients).
                4. Our AI model will predict the approximate yield of the crop under the specified conditions.

    

                #### Purpose of Our Project
                In recent years, India has faced numerous challenges in agriculture, leading to significant impacts on farmers and food security. 
                These challenges have highlighted the urgent need for innovative solutions to support our agricultural community. Motivated by the desire 
                to empower farmers and enhance their resilience against various adversities, we embarked on this project. Our goal is to provide effective 
                tools and resources that not only address current agricultural issues but also promote sustainable practices for the future.
                """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success(f"Model is Predicting it's a {class_name[result_index]}")

# Crop Yield Prediction - Additional Page
elif app_mode == "Crop Yield Prediction":
    st.image(yield_prediction_image, width=650)
    st.title('Yield Crop Prediction')
    html_temp = '''
    <div style='background-color:red; padding:12px'>
    <h1 style='color:  #000000; text-align: center;'>Yield Crop Prediction Machine Learning Model</h1>
    </div>
    <h2 style='color:  red; text-align: center;'>Please Enter Input</h2>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)
    
    country = st.selectbox("Type or Select a Country from the Dropdown.", df_main['area'].unique())
    crop = st.selectbox("Type or Select a Crop from the Dropdown.", df_main['item'].unique())
    average_rainfall = st.number_input('Enter Average Rainfall (mm-per-year).', value=None)
    presticides = st.number_input('Enter Pesticides per Tonnes Use (tonnes of active ingredients).', value=None)
    avg_temp = st.number_input('Enter Average Temperature (degree Celsius).', value=None)
    input_data = [country, crop, average_rainfall, presticides, avg_temp]
    
    if st.button('Predict'):
        if None in input_data or '' in input_data:
            st.warning("Please fill in all the fields.")
        else:
            result = prediction(input_data)
            temp = '''
            <div style='background-color:navy; padding:8px'>
            <h1 style='color: gold  ; text-align: center;'>{}</h1>
            </div>
            '''.format(result)
            st.markdown(temp, unsafe_allow_html=True)

# Pest Outbreak Prediction Page
elif app_mode == "Pest Outbreak Prediction":
    st.title("Pest Outbreak Chance Calculator")

    # Dropdown for seasons
    season = st.selectbox("Select Season:", ["Zaid", "Kharif", "Rabi"])

    # Dropdown for states
    states = [
        "Andhra Pradesh", "West Bengal", "Telangana", "Tamil Nadu", "Kerala",
        "Assam", "Punjab", "Haryana", "Odisha", "Gujarat", "Madhya Pradesh",
        "Bihar", "Uttar Pradesh", "Rajasthan", "Maharashtra"
    ]
    state = st.selectbox("Select State:", states)

    # Dropdown for crops
    crops = ["Rice", "Wheat", "Jowar", "Bajra"]
    crop = st.selectbox("Select Crop:", crops)

    # Define pest outbreak data based on previous information
    outbreak_data = {
        ("Kharif", "Andhra Pradesh", "Rice"): ("Rice Leaf Folder", "High"),
        ("Kharif", "West Bengal", "Rice"): ("Rice Leaf Folder", "High"),
        ("Kharif", "Telangana", "Rice"): ("Rice Leaf Folder", "High"),
        ("Kharif", "Tamil Nadu", "Rice"): ("Brown Planthopper", "Medium"),
        ("Kharif", "Kerala", "Rice"): ("Stem Borer", "Medium"),
        ("Kharif", "Assam", "Rice"): ("Stem Borer", "Medium"),
        ("Kharif", "Punjab", "Wheat"): ("Armyworm", "Low"),
        ("Rabi", "Haryana", "Wheat"): ("Aphids", "Medium"),
        ("Kharif", "Odisha", "Jowar"): ("Leaf Blight", "High"),
        ("Zaid", "Andhra Pradesh", "Rice"): ("Fall Armyworm", "Medium"),
    }

    # Check for pest outbreak
    if st.button("Check Pest Outbreak"):
        key = (season, state, crop)
        if key in outbreak_data:
            pest, chance = outbreak_data[key]
            st.success(f"Pest Outbreak Detected: {pest} with a chance of {chance}")
            
            # Defense strategies
            st.markdown("### Defense Strategies Against Pest Attacks:")
            st.markdown("- **Regular Monitoring:** Frequently inspect crops for signs of pest damage.")
            st.markdown("- **Integrated Pest Management (IPM):** Use a combination of biological controls, chemical treatments, and cultural practices.")
            st.markdown("- **Timely Action:** Implement measures as soon as signs of pests are detected.")
        else:
            st.warning("There are no pest outbreak chances for the selected season, state, and crop.")

# Farmer Support Schemes Page
elif app_mode == "Farmer Support Schemes":
    st.title("Farmer Support Schemes")

    # List of schemes with details, including actual hyperlinks
    schemes = {
      "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)": {
        "Objective": "Provides direct income support to small and marginal farmers.",
        "Benefits": "₹6,000 annually, disbursed in three equal installments.",
        "Apply": "[PM-KISAN Application](https://pmkisan.gov.in/)"
      },
      "Pradhan Mantri Fasal Bima Yojana (PMFBY)": {
        "Objective": "Offers insurance coverage for crop loss due to natural disasters, pests, and diseases.",
        "Benefits": "Insurance at low premiums for crop failure.",
        "Apply": "[PMFBY Application](https://pmfby.gov.in/)"
      },
      "Pradhan Mantri Krishi Sinchai Yojana (PMKSY)": {
        "Objective": "Focuses on enhancing irrigation facilities to ensure water availability.",
        "Benefits": "Assistance in building irrigation systems like drip and sprinkler irrigation.",
        "Apply": "[PMKSY Guidelines](https://pmksy.gov.in/)"
      },
      "Soil Health Card Scheme": {
        "Objective": "Provides farmers with soil health cards to assess the quality of their soil.",
        "Benefits": "Helps farmers use the appropriate fertilizers and nutrients.",
        "Apply": "[Soil Health Card Details](https://soilhealth.dac.gov.in/)"
      },
      "Kisan Credit Card (KCC) Scheme": {
        "Objective": "Provides short-term loans to farmers for agricultural expenses.",
        "Benefits": "Low-interest loans for farm-related expenses, including crop production.",
        "Apply": "[KCC Application](https://www.pmkisan.gov.in/KisanCreditCard.aspx)"
      },
      "National Agriculture Market (e-NAM)": {
        "Objective": "Provides a unified national market for agricultural commodities.",
        "Benefits": "Farmers can sell their produce through an online platform for better prices.",
        "Apply": "[e-NAM Portal](https://enam.gov.in/)"
      },
      "Paramparagat Krishi Vikas Yojana (PKVY)": {
        "Objective": "Promotes organic farming through financial support.",
        "Benefits": "Offers training and financial support for farmers who adopt organic farming.",
        "Apply": "[PKVY Details](https://agricoop.nic.in/en/pkvyguidelines)"
      },
      "Micro Irrigation Fund (MIF)": {
        "Objective": "Promotes efficient water usage in agriculture through micro-irrigation systems.",
        "Benefits": "Assistance for installing drip and sprinkler irrigation systems.",
        "Apply": "[MIF Guidelines](https://pmksy.gov.in/microIrrigation/)"
      },
      "Pradhan Mantri Kisan Maandhan Yojana": {
        "Objective": "Offers a pension scheme for small and marginal farmers.",
        "Benefits": "Farmers receive a pension of ₹3,000 per month after the age of 60.",
        "Apply": "[PM Kisan Maandhan Application](https://maandhan.in/)"
      },
      "Rashtriya Krishi Vikas Yojana (RKVY)": {
        "Objective": "Encourages state governments to invest in agriculture and related sectors.",
        "Benefits": "Funds for agricultural infrastructure, storage, and marketing.",
        "Apply": "[RKVY Guidelines](https://rkvy.nic.in/)"
      },
      "National Mission for Sustainable Agriculture (NMSA)": {
        "Objective": "Promotes sustainable agriculture practices to adapt to climate change.",
        "Benefits": "Focuses on improving soil health, water management, and climate resilience.",
        "Apply": "[NMSA Details](https://nmsa.dac.gov.in/)"
      },
      "National Food Security Mission (NFSM)": {
        "Objective": "Aims to increase the production of key crops like rice, wheat, and pulses.",
        "Benefits": "Assistance in crop improvement practices and production technologies.",
        "Apply": "[NFSM Guidelines](https://nfsm.gov.in/)"
      },
      "Pradhan Mantri Annadata Aay Sanrakshan Abhiyan (PM-AASHA)": {
        "Objective": "Ensures minimum support prices (MSP) to farmers for their produce.",
        "Benefits": "Provides price assurance and procurement of produce at guaranteed prices.",
        "Apply": "[PM-AASHA Details](https://agricoop.nic.in/en/pm-aasha)"
      },
      "Gramin Bhandaran Yojana": {
        "Objective": "Provides storage facilities for agricultural produce.",
        "Benefits": "Offers financial assistance for building rural godowns to store produce.",
        "Apply": "[Gramin Bhandaran Guidelines](https://nhb.gov.in/)"
      },
      "National Mission on Oilseeds and Oil Palm (NMOOP)": {
        "Objective": "Aims to increase the production of oilseeds and the area under oil palm.",
        "Benefits": "Offers subsidies for oilseed cultivation and processing units.",
        "Apply": "[NMOOP Guidelines](https://nmoop.gov.in/)"
      },
      "Sub-Mission on Agricultural Mechanization (SMAM)": {
        "Objective": "Promotes the use of mechanized farming techniques.",
        "Benefits": "Provides subsidies for purchasing agricultural machinery and tools.",
        "Apply": "[SMAM Guidelines](https://agricoop.nic.in/schemes/smam)"
      },
      "Dairy Entrepreneurship Development Scheme (DEDS)": {
        "Objective": "Supports dairy farming and encourages rural farmers to adopt dairy as an additional income source.",
        "Benefits": "Financial aid for setting up dairy farms and buying dairy equipment.",
        "Apply": "[DEDS Guidelines](https://nabard.org/)"
      },
      "Pradhan Mantri Kaushal Vikas Yojana (PMKVY) for Agriculture": {
        "Objective": "Provides skill development training to rural youth in various agricultural activities.",
        "Benefits": "Training for different agricultural practices like crop production, irrigation, and livestock management.",
        "Apply": "[PMKVY Application](https://pmkvyofficial.org/)"
      },
      "Krishi Vigyan Kendras (KVKs)": {
        "Objective": "Provides on-the-ground training and demonstrations for farmers in various agricultural techniques.",
        "Benefits": "Farmers receive practical knowledge and can implement the latest technologies in agriculture.",
        "Apply": "[KVK Details](https://kvk.icar.gov.in/)"
      },
      "Agri-Clinics and Agri-Business Centers Scheme (ACABC)": {
        "Objective": "Supports agricultural graduates in setting up their own agri-business centers.",
        "Benefits": "Provides credit-linked back-ended capital subsidy and training to start agribusiness.",
        "Apply": "[ACABC Details](https://agricoop.nic.in/en/acabc)"
      }
    }

    # Dropdown for selecting a scheme
    selected_scheme = st.selectbox("Select a scheme to view details:", list(schemes.keys()))

    # Displaying the selected scheme's details
    st.subheader(selected_scheme)
    st.write(f"**Objective:** {schemes[selected_scheme]['Objective']}")
    st.write(f"**Benefits:** {schemes[selected_scheme]['Benefits']}")
    st.write(f"**Apply Here:** {schemes[selected_scheme]['Apply']}")