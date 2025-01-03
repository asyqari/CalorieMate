import streamlit as st
import tensorflow as tf
import numpy as np
import requests

# Edamam API credentials
EDAMAM_APP_ID = "218d4d89"
EDAMAM_APP_KEY = "33476bccb9a1e3d824e619d255616993"


@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("trained_model_new.h5")
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


model = load_model()


# Logic Prediction
def predict_img(img):
    if model is None:
        return None

    image = tf.keras.preprocessing.image.load_img(img, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)

    # logging.info(f"Predicted index: {predicted_index}")
    # logging.info(f"Predicted label: {class_names[predicted_index]}")
    return np.argmax(prediction)


# Calculate nutrients and tips using Edamam API
def calculate_nutrients_and_tips(food_item):
    try:
        url = f"https://api.edamam.com/api/food-database/v2/parser"
        params = {
            "ingr": f"{food_item}",
            "app_id": EDAMAM_APP_ID,
            "app_key": EDAMAM_APP_KEY,
        }
        response = requests.get(url, params=params)
        data = response.json()
        nutrients = data["hints"][0]["food"]["nutrients"]

        calories = nutrients.get("ENERC_KCAL", 0)
        protein = nutrients.get("PROCNT", 0)
        fat = nutrients.get("FAT", 0)
        carbs = nutrients.get("CHOCDF", 0)
        fiber = nutrients.get("FIBTG", 0)

        # Calculate tips for various activities
        activities = {
            "walking": calories * 10,  # 10 min per kcal
            "running": calories * 7,  # 7 min per kcal
            "cycling": calories * 8,  # 8 min per kcal
            "swimming": calories * 9,  # 9 min per kcal
        }

        tips = (
            f"To burn the calories in 100g of {food_item}, you could:\n"
            f"- Walk for approximately {int(activities['walking'])} minutes.\n"
            f"- Run for approximately {int(activities['running'])} minutes.\n"
            f"- Cycle for approximately {int(activities['cycling'])} minutes.\n"
            f"- Swim for approximately {int(activities['swimming'])} minutes."
        )

        return calories, protein, fat, carbs, fiber, tips
    except Exception as e:
        print(f"Error getting nutrient information: {e}")
        return None, None, None, None, None, None


# Sidebar
st.title("CalorieMate")
st.sidebar.title("Dashboard")
page_selected = st.sidebar.selectbox("Select Page", ["Abouts", "Predict"])

# About page
if page_selected == "Abouts":
    st.header("Abouts", divider="rainbow")
    st.subheader("About Project")
    long_text = """This project is a tool to find out the nutrients contained in a food and also its calories. retrieve nutritional data from each existing food 
    Apart from that, there will be tips on how to burn these calories. 
    This project uses a CNN model to classify images and then uses the API from Edamam to automatically."""

    st.markdown(
        f'<div style="max-height: 300px; overflow-y: auto; padding: 10px; border: 1px solid #ddd;">{long_text}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("About Dataset")
    st.text("Context: ")
    st.markdown(
        "Link to the Dataset: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition"
    )
    st.markdown(
        "This dataset encompasses images of various fruits and vegetables, providing a diverse collection for image recognition tasks. The included food items are:"
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            - **Fruits:**
                - Banana
                - Apple
                - Pear
                - Grapes
                - Orange
                - Kiwi
                - Watermelon
                - Pomegranate
                - Pineapple
                - Mango
            """
        )

    with col2:
        st.markdown(
            """
            - **Vegetables:**
                - Cucumber
                - Carrot
                - Capsicum
                - Onion
                - Potato
                - Lemon
                - Tomato
                - Radish
                - Beetroot
                - Cabbage
                - Lettuce
                - Spinach
                - Soybean
            """
        )

    with col3:
        st.markdown(
            """
            - **More Vegetables:**
                - Turnip
                - Corn
                - Sweetcorn
                - Sweet Potato
                - Paprika
                - Jalapeño
                - Ginger
                - Garlic
                - Peas
                - Eggplant
                - Cauliflower
                - Bell Pepper
                - Chilli Pepper
            """
        )

    st.subheader("How It Works")
    long_text = """First you upload an image of food you want, and then the model will give out the prediction of the image, and then it will also give you some nutrition facts about the food.
    Lastly it will give you some tips to burn the calories that the food contains."""

    st.markdown(
        f'<div style="max-height: 300px; overflow-y: auto; padding: 10px; border: 1px solid #ddd;">{long_text}</div>',
        unsafe_allow_html=True,
    )
    st.subheader("The Model")
    long_text = """The model used uses the concept of a Convolutional Neural Network and a dataset from Kaggle in the form of fruit and vegetable image data.
    The model was built with multilayers and achieved an accuracy of 92 percent (can increase if tuned and retrained)"""

    st.markdown(
        f'<div style="max-height: 300px; overflow-y: auto; padding: 10px; border: 1px solid #ddd;">{long_text}</div>',
        unsafe_allow_html=True,
    )


# Prediction Page
if page_selected == "Predict":
    st.header("Prediction Page", divider="rainbow")
    st.subheader("Step To Use:")
    st.text("1. Upload Image")
    img = st.file_uploader("Select your desired image")
    st.text("2. Press the show to show the image")
    if st.button("Show image"):
        if img is not None:
            st.image(img)

    st.text("3. Press the predict to start the process")
    if st.button("Predict"):
        result_id = predict_img(img)
        if result_id is not None:
            # Read the labels file
            with open("labels.txt") as f:
                content = f.readlines()
            labels = [x.strip() for x in content]

            predicted_food = labels[result_id]
            st.success(f"The model predicted that this is a {predicted_food}")

            # Calculate nutrients and tips
            calories, protein, fat, carbs, fiber, tips = calculate_nutrients_and_tips(
                predicted_food.lower()
            )

            if calories is not None:
                st.markdown(f"### Nutritional Information for 100g of {predicted_food}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Calories**: {calories} kcal")
                    st.markdown(f"**Protein**: {protein} g")
                with col2:
                    st.markdown(f"**Fat**: {fat} g")
                    st.markdown(f"**Carbohydrates**: {carbs} g")
                    st.markdown(f"**Fiber**: {fiber} g")

                st.markdown("### Tips")
                st.info(tips)
            else:
                st.warning("Nutrient information not available for this food item.")
        else:
            st.error("Error predicting the image. Please try again.")
