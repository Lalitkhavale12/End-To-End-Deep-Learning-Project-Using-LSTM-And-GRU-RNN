import streamlit as st
import numpy as np
import pickle
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# FILE PATHS
# ============================================================

# Get the folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "next_word_lstm.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pickle"


# ============================================================
# LOAD MODEL AND TOKENIZER
# ============================================================

@st.cache_resource
def load_resources():

    # Check whether model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    # Check whether tokenizer exists
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {TOKENIZER_PATH}"
        )

    # Load trained LSTM model
    model = load_model(MODEL_PATH)

    # Load tokenizer
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)

    return model, tokenizer


# Load resources
model, tokenizer = load_resources()


# ============================================================
# CREATE REVERSE WORD INDEX
# ============================================================

# Converts:
# word -> index
#
# Into:
# index -> word

index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
}


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(text, tokenizer, max_sequence_len):

    # Convert input text into numerical tokens
    token_list = tokenizer.texts_to_sequences([text])[0]

    # If all words are unknown
    if not token_list:
        return None

    # Keep only the last required words
    token_list = token_list[-(max_sequence_len - 1):]

    # Pad sequence
    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    return token_list


# ============================================================
# PREDICT NEXT WORD
# ============================================================

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

    # Check invalid input
    if token_list is None:
        return []

    # Predict probabilities
    prediction = model.predict(
        token_list,
        verbose=0
    )[0]

    # Get top K indexes
    top_indices = np.argsort(
        prediction
    )[-top_k:][::-1]

    results = []

    # Convert indexes to words
    for index in top_indices:

        word = index_to_word.get(int(index))

        if word:

            confidence = float(
                prediction[index] * 100
            )

            results.append({
                "word": word,
                "confidence": confidence
            })

    return results


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🤖 Next Word Prediction")

st.write(
    "Enter a sequence of words and let the LSTM model "
    "predict the most likely next word."
)

st.divider()


# ============================================================
# INPUT
# ============================================================

input_text = st.text_input(
    "Enter your sentence",
    value="To be or not to",
    placeholder="Example: To be or not to"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

max_sequence_len = model.input_shape[1] + 1

with st.expander("ℹ️ Model Information"):

    st.write(
        f"**Vocabulary Size:** "
        f"{len(tokenizer.word_index)}"
    )

    st.write(
        f"**Maximum Input Length:** "
        f"{max_sequence_len - 1} words"
    )

    st.write(
        "**Model Type:** LSTM"
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 Predict Next Word",
    use_container_width=True
):

    # Validate input
    if not input_text.strip():

        st.warning(
            "⚠️ Please enter some text."
        )

    else:

        # Show loading animation
        with st.spinner("🤖 Predicting the next word..."):

            predictions = predict_next_word(
                model=model,
                tokenizer=tokenizer,
                index_to_word=index_to_word,
                text=input_text,
                max_sequence_len=max_sequence_len,
                top_k=5
            )

        # ====================================================
        # SHOW RESULTS
        # ====================================================

        if predictions:

            best_prediction = predictions[0]

            st.success(
                "Prediction Complete!"
            )

            st.subheader(
                "🎯 Predicted Next Word"
            )

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


            # =================================================
            # COMPLETE SENTENCE
            # =================================================

            st.subheader(
                "📝 Complete Sentence"
            )

            st.markdown(
                f"""
                **{input_text}**
                
                ➡️ **{best_prediction['word']}**
                """
            )


            # =================================================
            # TOP 5 PREDICTIONS
            # =================================================

            st.subheader(
                "📊 Top Predictions"
            )

            for i, prediction in enumerate(
                predictions,
                start=1
            ):

                word = prediction["word"]
                confidence = prediction["confidence"]

                st.write(
                    f"**{i}. {word}** "
                    f"— {confidence:.2f}%"
                )

                st.progress(
                    min(
                        int(confidence),
                        100
                    )
                )

        else:

            st.error(
                "❌ Unable to predict a word. "
                "Try using words from the training dataset."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Streamlit • TensorFlow • LSTM"
)