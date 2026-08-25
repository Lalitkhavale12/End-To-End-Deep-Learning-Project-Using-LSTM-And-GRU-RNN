import streamlit as st  # noqa: I001
import numpy as np
import pickle

from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🤖",
    layout="centered"
)


# ---------------------------------------------------
# LOAD MODEL AND TOKENIZER
# ---------------------------------------------------

@st.cache_resource
def load_resources():
    
    model = load_model("next_word_lstm.h5")
    
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    
    return model, tokenizer


model, tokenizer = load_resources()


# ---------------------------------------------------
# CREATE REVERSE WORD INDEX
# ---------------------------------------------------

# Original:
# tokenizer.word_index
#
# Example:
# {
#     "to": 1,
#     "be": 2,
#     "or": 3
# }
#
# Reverse dictionary:
# {
#     1: "to",
#     2: "be",
#     3: "or"
# }

index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
}


# ---------------------------------------------------
# FUNCTION TO PREPARE INPUT
# ---------------------------------------------------

def prepare_input(text, tokenizer, max_sequence_len):
    
    # Convert text into token IDs
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Keep only the required number of words
    token_list = token_list[-(max_sequence_len - 1):]
    
    # Pad sequence if required
    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )
    
    return token_list


# ---------------------------------------------------
# FUNCTION TO PREDICT NEXT WORD
# ---------------------------------------------------

def predict_next_word(
    model,
    tokenizer,
    index_to_word,
    text,
    max_sequence_len,
    top_k=5
):
    
    # Prepare input
    token_list = prepare_input(
        text,
        tokenizer,
        max_sequence_len
    )
    
    # Make prediction
    prediction = model.predict(
        token_list,
        verbose=0
    )[0]
    
    # Get top K predictions
    top_indices = np.argsort(prediction)[-top_k:][::-1]
    
    results = []
    
    for index in top_indices:
        
        word = index_to_word.get(index)
        confidence = prediction[index] * 100
        
        if word:
            results.append(
                {
                    "word": word,
                    "confidence": confidence
                }
            )
    
    return results


# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------

st.title("🤖 Next Word Prediction")
st.caption("Predict the next word using an LSTM Neural Network")

st.divider()


# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

st.subheader("Enter a sentence")

input_text = st.text_input(
    label="Input Text",
    value="To be or not to",
    placeholder="Type a sentence..."
)


# ---------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------

max_sequence_len = model.input_shape[1] + 1

with st.expander("ℹ️ Model Information"):
    
    st.write(f"**Vocabulary Size:** {len(tokenizer.word_index)}")
    
    st.write(
        f"**Maximum Input Length:** "
        f"{max_sequence_len - 1} words"
    )
    
    st.write("**Model Type:** LSTM")


# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------

if st.button(
    "🔮 Predict Next Word",
    use_container_width=True
):
    
    # Validate input
    if not input_text.strip():
        
        st.warning(
            "⚠️ Please enter some text before predicting."
        )
    
    else:
        
        with st.spinner("🤖 Predicting..."):
            
            predictions = predict_next_word(
                model=model,
                tokenizer=tokenizer,
                index_to_word=index_to_word,
                text=input_text,
                max_sequence_len=max_sequence_len,
                top_k=5
            )
        
        if predictions:
            
            best_prediction = predictions[0]
            
            # ---------------------------------------------------
            # BEST PREDICTION
            # ---------------------------------------------------
            
            st.success("Prediction Complete!")
            
            st.subheader("🎯 Predicted Next Word")
            
            col1, col2 = st.columns(2)
            
            with col1:
                
                st.metric(
                    "Next Word",
                    best_prediction["word"]
                )
            
            with col2:
                
                st.metric(
                    "Confidence",
                    f"{best_prediction['confidence']:.2f}%"
                )
            
            
            # ---------------------------------------------------
            # COMPLETE SENTENCE
            # ---------------------------------------------------
            
            st.subheader("📝 Complete Prediction")
            
            completed_sentence = (
                f"{input_text} "
                f"**{best_prediction['word']}**"
            )
            
            st.markdown(completed_sentence)
            
            
            # ---------------------------------------------------
            # TOP 5 PREDICTIONS
            # ---------------------------------------------------
            
            st.subheader("📊 Top Predictions")
            
            for i, prediction in enumerate(predictions, start=1):
                
                word = prediction["word"]
                confidence = prediction["confidence"]
                
                st.write(
                    f"**{i}. {word}** — "
                    f"{confidence:.2f}%"
                )
                
                st.progress(
                    min(int(confidence), 100)
                )
        
        else:
            
            st.error(
                "❌ Unable to predict a word. "
                "Please try a different sentence."
            )


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
    "Built with Streamlit | TensorFlow | LSTM"
)