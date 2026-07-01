import streamlit as st
import joblib

# load the model - using cache so it doesnt reload every time the page refreshes
@st.cache_data
def load_model():
    model = joblib.load("sentiment_model.pkl")
    return model

model = load_model()

# app title and description
st.title("So... Was It Actually Good?")
st.write("Let's be honest — not every movie deserves five stars. Paste a review below and I'll tell you whether you loved it or suffered through it.")
st.write("No sugarcoating. The model has spoken.")

# text box for the user to type their review
user_input = st.text_area("Drop your review here and let's find out:")

# button to trigger the prediction
if st.button("Tell me the truth"):
    # make sure the user actually typed something
    if user_input == "":
        st.warning("umm... you forgot to actually write something.")
    else:
        # predict - has to be in a list because the model expects multiple inputs
        prediction = model.predict([user_input])[0]
        probability = model.predict_proba([user_input])[0]

        # show the result
        st.subheader("The verdict:")

        if prediction == "positive":
            st.success("Looks like you actually enjoyed this one! Positive vibes detected.")
        else:
            st.error("Yikes. The model can feel your pain. Definitely a negative one.")

        # show the probabilities too (bonus part)
        st.write("How confident are we?")
        st.write(f"Positive: {round(probability[1] * 100, 2)}%")
        st.write(f"Negative: {round(probability[0] * 100, 2)}%")
