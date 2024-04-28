import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_csv('fertilizer_recommendations.csv')

#Tensorflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_plant_disease_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition","Soil Nutrients"])



#Main Page
class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                      'Blueberry__healthy', 'Cherry(including_sour)___Powdery_mildew',
                      'Cherry_(including_sour)__healthy', 'Corn(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                      'Corn_(maize)__Common_rust', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn(maize)___healthy',
                      'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
                      'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy',
                      'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy',
                      'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew',
                      'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot',
                      'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold',
                      'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
                      'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                      'Tomato___healthy']
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Leaf Disease Recognition and Fertilizer recommendation system! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently and also recommeding solutions on the detected plant leaf disease. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)


#Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")

    if st.button("Show Image"):
        if test_image:
            st.image(test_image, width=4, use_column_width=True)
        else:
            st.warning("Please select the image",icon="⚠️")
    
    if st.button("Predict"):
        if test_image:
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            

            predicted_disease = class_name[result_index]
            st.success("Model has predicted : It's a {}".format(predicted_disease))
        
            recommended_fertilizer_values = df['Recommended Fertilizer'].tolist()
            
        # Print the array of recommended fertilizer values
        # recommended_fertilizer_values
            st.write("Our recommendation")
            Recommended_Fertilizer = recommended_fertilizer_values[result_index]
            st.success("Recommended Fertilizer : {}".format(Recommended_Fertilizer))
        else:
            st.warning("No image is selected yet to predict result.",icon="⚠️")

elif (app_mode == "Soil Nutrients"):
    ideal_soil_fertility_levels = {
                'Low': {'N': 0.1, 'P': 0.05, 'K': 0.1},
                'Medium': {'N': 0.2, 'P': 0.1, 'K': 0.2},
                'High': {'N': 0.3, 'P': 0.15, 'K': 0.3}
                }
    st.title("Ideal Soil Fertility Levels : ")
    for level, values in ideal_soil_fertility_levels.items():
        formatted_values = ", ".join([f"{key}: {value}" for key, value in values.items()])
        st.write(f"- {level}: {formatted_values}")   
        
        
    soil_fertility_levels = {
            'Apple__Apple_scab': {'N': 0.2, 'P': 0.1, 'K': 0.2},  
            'Apple_Black_rot': {'N': 0.3, 'P': 0.15, 'K': 0.3},   
            'Apple_Cedar_apple_rust': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Apple__healthy': {'N': 0.2, 'P': 0.1, 'K': 0.2},  
            'Blueberry__healthy': {'N': 0.2, 'P': 0.1, 'K': 0.2},  
            'Cherry(including_sour)___Powdery_mildew': {'N': 0.25, 'P': 0.12, 'K': 0.25}, 
            'Cherry_(including_sour)__healthy': {'N': 0.2, 'P': 0.1, 'K': 0.2},  
            'Corn(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'N': 0.25, 'P': 0.12, 'K': 0.25}, 
            'Corn_(maize)__Common_rust': {'N': 0.3, 'P': 0.15, 'K': 0.3},  
            'Corn_(maize)__Northern_Leaf_Blight': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Corn(maize)___healthy': {'N': 0.2, 'P': 0.1, 'K': 0.2},  
            'Grape__Black_rot': {'N': 0.3, 'P': 0.15, 'K': 0.3},  
            'Grape_Esca(Black_Measles)': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Grape__healthy': {'N': 0.2, 'P': 0.1, 'K': 0.2},  
            'Orange_Haunglongbing(Citrus_greening)': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Peach___Bacterial_spot': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Peach__healthy': {'N': 0.2, 'P': 0.1, 'K': 0.2},  
            'Pepper,_bell_Bacterial_spot': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Pepper,bell_healthy': {'N': 0.2, 'P': 0.1, 'K': 0.2},  
            'Potato__Early_blight': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Potato_Late_blight': {'N': 0.3, 'P': 0.15, 'K': 0.3},  
            'Potato__healthy': {'N': 0.2, 'P': 0.1, 'K': 0.2},  
            'Raspberry__healthy': {'N': 0.2, 'P': 0.1, 'K': 0.2},  
            'Soybean_healthy': {'N': 0.2, 'P': 0.1, 'K': 0.2},  
            'Squash__Powdery_mildew': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Strawberry__Leaf_scorch': {'N': 0.2, 'P': 0.1, 'K': 0.2},  
            'Strawberry_healthy': {'N': 0.2, 'P': 0.1, 'K': 0.2},  
            'Tomato__Bacterial_spot': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Tomato__Early_blight': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Tomato_Late_blight': {'N': 0.3, 'P': 0.15, 'K': 0.3},  
            'Tomato__Leaf_Mold': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Tomato__Septoria_leaf_spot': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Tomato__Spider_mites Two-spotted_spider_mite': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Tomato__Target_Spot': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Tomato_Tomato_Yellow_Leaf_Curl_Virus': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Tomato__Tomato_mosaic_virus': {'N': 0.25, 'P': 0.12, 'K': 0.25},  
            'Tomato___healthy': {'N': 0.2, 'P': 0.1, 'K': 0.2},  
        }
    
    
    st.header("Enter soil nutrients in above range : ")
    # Input fields for NPK values
    n_input = st.text_input("Enter N value:")
    p_input = st.text_input("Enter P value:")
    k_input = st.text_input("Enter K value:")

    # Save the input values
    if st.button("Save"):
        npk_values = {
            'N': float(n_input),
            'P': float(p_input),
            'K': float(k_input)
        }
        st.write("NPK values saved:", npk_values)
    
        
    
    test_image = st.file_uploader("Choose an Image:")
    

    if st.button("Predict"):
    # class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    #               'Blueberry__healthy', 'Cherry(including_sour)___Powdery_mildew',
    #               'Cherry_(including_sour)__healthy', 'Corn(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    #               'Corn_(maize)__Common_rust', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn(maize)___healthy',
    #               'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
    #               'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
    #               'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy',
    #               'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy',
    #               'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew',
    #               'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot',
    #               'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold',
    #               'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
    #               'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    #               'Tomato___healthy']
        st.write("Our Prediction : ")
    
        result_index = model_prediction(test_image)
        predicted_disease = class_name[result_index]
        predicted_disease
    
    # Get the ideal NPK values for the predicted disease from soil_fertility_levels
        ideal_values = soil_fertility_levels.get(predicted_disease, None)
#ideal_values

# Retrieve the user-input NPK values
        npk_values = {
            'N': float(n_input),
            'P': float(p_input),
            'K': float(k_input)
        }

        # Compare each nutrient value with the corresponding ideal range
        comparison_labels = {}
        if ideal_values:
            n_user_value = npk_values.get('N')
            p_user_value = npk_values.get('P')
            k_user_value = npk_values.get('K')
            
            n_ideal_value = ideal_values.get('N')
            p_ideal_value = ideal_values.get('P')
            k_ideal_value = ideal_values.get('K') 
            
            if n_user_value is not None:
                if n_user_value < n_ideal_value * 0.8:
                    st.write("*Nitrogen level in your soil is too low. Ideal is*",n_ideal_value)
                    st.write("Here are some ways to increase nitrogen levels in the soil:")
                    st.write("- Use nitrogen-rich fertilizers: Fertilizers containing ammonium nitrate, urea, or ammonium sulfate can increase nitrogen levels in the soil.")
                    st.write("- Apply organic matter: Organic materials like compost, manure, or mulch can improve soil fertility and increase nitrogen levels over time.")
                    st.write("- Plant nitrogen-fixing cover crops: Leguminous cover crops such as clover, peas, or beans can fix atmospheric nitrogen into the soil through a symbiotic relationship with nitrogen-fixing bacteria.")
                elif n_user_value <= n_ideal_value * 1.2:
                    st.write("*Nitrogen level in your soil  is good.*")
                else:
                    st.write("*Nitrogen level in your soil is too high. Ideal is*",n_ideal_value)

            if p_user_value is not None:
                if p_user_value < p_ideal_value * 0.8:
                    st.write("*Phosprous level in your soil is too low. Ideal is*",p_ideal_value)
                    st.write("Here are some ways to increase Phosphorus levels in the soil:")
                    st.write("- Use phosphorus-rich fertilizers: Fertilizers containing phosphate rock, bone meal, or superphosphate can increase phosphorus levels in the soil.")
                    st.write("- Apply organic sources: Organic materials like bone meal, fish meal, or composted manure can provide slow-release phosphorus to the soil.")
                    st.write("- Adjust pH levels: Maintaining soil pH between 6.0 and 7.0 optimizes phosphorus availability in the soil.")
                elif p_user_value <= p_ideal_value * 1.2 :
                    st.write("*Phosprous level in your soil is good.*")
                else:
                    st.write("*Phosprous level in your soil is too high. Ideal is*",p_ideal_value)
                    
            if k_user_value is not None:
                if k_user_value < k_ideal_value * 0.8:
                    st.write("*Potassium level in your soil is too low. Ideal is*",k_ideal_value)
                    st.write("Here are some ways to increase Potassium levels in the soil:")
                    st.write("- Use potassium-rich fertilizers: Fertilizers containing potassium sulfate, potassium chloride, or potassium nitrate can increase potassium levels in the soil.")
                    st.write("- Apply potassium-rich amendments: Materials like wood ash, granite dust, or kelp meal can provide potassium to the soil.")
                    st.write("- Use potash: Potassium-rich salts like muriate of potash or sulfate of potash can be applied to increase potassium levels.")
                elif k_user_value <= k_ideal_value * 1.2:
                    st.write("*Potassium level in your soil is good.*")
                else:
                    st.write("*Potassium level in your soil is too high. Ideal is*",k_ideal_value)        
            
            if n_user_value > n_ideal_value * 1.2 or p_user_value > p_ideal_value * 1.2 or k_user_value > k_ideal_value * 1.2 :
                st.write("*For Decreasing nutrients levels, do following :* ")
                st.write("- Dilution: In cases where nutrient levels are too high, diluting the soil with clean soil or compost can reduce nutrient concentrations.")
                st.write("- Leaching: Watering the soil heavily can help leach excess nutrients deeper into the soil profile, away from the root zone.")
                st.write("- Selective fertilization: Limiting the use of fertilizers rich in the specific nutrient can help prevent further increases in soil nutrient levels.")

