import streamlit as st
import pandas as pd
import numpy as np
import joblib

pipe_lr = joblib.load(open(r'emotion_classifier_pipe_24_oct_2023.pkl', 'rb'))

def predict_emotion(docx):
    result = pipe_lr.predict([docx])
    return result[0]

def get_prediction_proba(docx):
    result = pipe_lr.predict_proba([docx])
    return result

# emotions_emoji_dict = {'anger':"", 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'shame',
#        'surprise'}
def main():
    st.title('Emotion Recognition using text-classification (Mini project)')
    menu = ['Home', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader("Home")
        with st.form(key='emotion_key'):
            raw_text = st.text_area('Enter text here')
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            # Apply functions here
            prediction = predict_emotion(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original text")
                st.write(raw_text)

                st.success("Prediction")
                # emoji_icon = emotions_emoji_dict[prediction]

                st.write(prediction)

            with col2:
                st.success('Prediction probability')
                st.write(probability)

    # elif choice == "Monitor":
    #     st.subheader("Monitor App")
    else:
        st.subheader("About")
        st.text('This app about text classification using scikit-learn library. I submitted this as a mini project.')

if __name__ == '__main__':
    main()
