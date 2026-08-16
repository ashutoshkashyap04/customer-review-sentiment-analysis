import streamlit as st
import requests


# Page configuration
st.set_page_config(
    page_title="Customer Review Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)


# Title
st.title("💬 Customer Review Sentiment Analysis")
st.write(
    "Enter a customer review below to predict its sentiment "
    "using a deep learning model."
)

API_URL = "http://127.0.0.1:8000/predict"

# review input box
review = st.text_area(
    "Enter your review",
    placeholder= "Example: The product quality is excellent and I am very satisfied! ",
    height = 150        #height of text area in pixels
)

# Prediction button
if st.button(
    "Analyze Sentiment",
    use_container_width= True
):
    if not review.strip():
        st.warning("Please enter a review before analyzing.")
        
    else:
        try:
            #Send request to FastAPI backend
            response = requests.post(
                API_URL,
                json= {"review": review}
            )
            
            #Check response
            if response.status_code == 200:
                result = response.json()
                
                rating = result['rating']
                confidence = result['confidence']
                
                # Display result
                st.success("Prediction completed successfully!")
                
                st.subheader("Prediction")
                
                st.write(f"**Rating:** {rating}⭐")
                st.write(f"**Confidence:** {confidence:.2f}")
                
            else:
                st.error(
                    f"Prediction failed!"
                    f"status code {response.status_code}."
                )
                
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the backend API."
                "Make sure the FastAPI server is running"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {e}")