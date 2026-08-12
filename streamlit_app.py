# streamlit_app.py - Web Interface for Emotion Recognition
"""
Interactive web interface using Streamlit
Run with: streamlit run streamlit_app.py
"""

# import streamlit as st
# import torch
# from transformers import AutoTokenizer
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime
# import os

# # Import your model classes
# from model.bert_baseline import BERTEmotionClassifier
# from config import Config
import streamlit as st
import torch
from transformers import AutoTokenizer
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Import DialogueRNN model
from model.dialogue_rnn import BERTDialogueRNN
from config import Config

# Page configuration
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

# Custom CSS
st.markdown(""" 
<style>
    .emotion-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .joy { background-color: #FFE5B4; }
    .sadness { background-color: #B4D4FF; }
    .anger { background-color: #FFB4B4; }
    .fear { background-color: #E5B4FF; }
    .surprise { background-color: #B4FFE5; }
    .disgust { background-color: #D4D4D4; }
    .neutral { background-color: #F0F0F0; }
</style>
""", unsafe_allow_html=True)

# ...existing page config and CSS...

@st.cache_resource
def load_model():
    """Load the trained DialogueRNN model (cached)"""
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    model = BERTDialogueRNN(config)
    model_path = 'outputs/dialogue_rnn_best_model.pth'
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, tokenizer, config
    else:
        st.error(f"Model not found at {model_path}")
        return None, None, None

def predict_emotion(text, model, tokenizer, config):
    """Predict emotion for given text using DialogueRNN"""
    # DialogueRNN expects batch of utterances (seq_len=1 for single text)
    inputs = tokenizer(
        [text],  # list of utterances
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=config.MAX_LENGTH
    )
    input_ids = inputs['input_ids'].unsqueeze(0)         # [batch=1, seq_len=1, max_length]
    attention_mask = inputs['attention_mask'].unsqueeze(0) # [batch=1, seq_len=1, max_length]
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)        # [batch=1, seq_len=1, num_classes]
        probabilities = torch.softmax(outputs.view(-1, outputs.size(-1)), dim=1)
        prediction = outputs.view(-1, outputs.size(-1)).argmax(dim=1).item()
        confidence = probabilities[0, prediction].item()
    
    all_probs = {
        config.EMOTIONS[i]: float(probabilities[0, i])
        for i in range(len(config.EMOTIONS))
    }
    
    return config.EMOTIONS[prediction], confidence, all_probs

# ...rest of your code remains unchanged...
# @st.cache_resource
# def load_model():
#     """Load the trained model (cached)"""
#     config = Config()
#     tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
#     model = BERTEmotionClassifier(config)
#     model_path = 'outputs/bert_baseline_best_model.pth'
    
#     if os.path.exists(model_path):
#         checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.eval()
#         return model, tokenizer, config
#     else:
#         st.error(f"Model not found at {model_path}")
#         return None, None, None

# def predict_emotion(text, model, tokenizer, config):
#     """Predict emotion for given text"""
#     inputs = tokenizer(
#         text,
#         return_tensors='pt',
#         truncation=True,
#         padding='max_length',
#         max_length=config.MAX_LENGTH
#     )
    
#     with torch.no_grad():
#         outputs = model(inputs['input_ids'], inputs['attention_mask'])
#         probabilities = torch.softmax(outputs, dim=1)
#         prediction = outputs.argmax(dim=1).item()
#         confidence = probabilities[0, prediction].item()
    
#     all_probs = {
#         config.EMOTIONS[i]: float(probabilities[0, i])
#         for i in range(len(config.EMOTIONS))
#     }
    
#     return config.EMOTIONS[prediction], confidence, all_probs

def get_emotion_emoji(emotion):
    """Get emoji for emotion"""
    emoji_map = {
        'joy': '😊', 'sadness': '😢', 'anger': '😠',
        'fear': '😨', 'surprise': '😮', 'disgust': '🤢',
        'neutral': '😐'
    }
    return emoji_map.get(emotion, '🤔')

# Main app
def main():
    st.title("🎭 Emotion Recognition ")
    st.markdown("Analyze emotions in text ")
    
    # Load model
    model, tokenizer, config = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("ℹ️ About")
    st.sidebar.info(
        "This app uses a BERT-based model to detect emotions in text. "
        "It can identify 7 emotions: joy, sadness, anger, fear, surprise, disgust, and neutral."
    )
    
    # Mode selection
    mode = st.radio(
        "Select Mode",
        ["Single Text", "Conversation Analysis"],
        horizontal=True
    )
    
    if mode == "Single Text":
        st.header("Single Text Analysis")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste your text here...",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        if analyze_button and text_input:
            with st.spinner("Analyzing..."):
                emotion, confidence, all_probs = predict_emotion(text_input, model, tokenizer, config)
            
            # Display results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="emotion-box {emotion}">
                    <h1>{get_emotion_emoji(emotion)}</h1>
                    <h2>{emotion.upper()}</h2>
                    <h3>{confidence:.1%} confident</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Create probability chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(all_probs.values()),
                        y=list(all_probs.keys()),
                        orientation='h',
                        marker_color=['#FFE5B4', '#B4D4FF', '#FFB4B4', '#E5B4FF', '#B4FFE5', '#D4D4D4', '#F0F0F0']
                    )
                ])
                fig.update_layout(
                    title="Emotion Probabilities",
                    xaxis_title="Probability",
                    yaxis_title="Emotion",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif mode == "Conversation Analysis":
        st.header("Conversation Analysis")
        
        # Number of utterances
        num_utterances = st.slider("Number of utterances:", 2, 10, 3)
        
        utterances = []
        speakers = []
        
        # Create input fields
        for i in range(num_utterances):
            col1, col2 = st.columns([1, 3])
            with col1:
                speaker = st.text_input(f"Speaker {i+1}:", value=f"Person {chr(65+i)}", key=f"speaker_{i}")
                speakers.append(speaker)
            with col2:
                utterance = st.text_input(f"Utterance {i+1}:", key=f"utterance_{i}")
                utterances.append(utterance)
        
        if st.button("🎭 Analyze Conversation", type="primary"):
            if all(utterances):
                results = []
                emotions = []
                
                with st.spinner("Analyzing conversation..."):
                    for i, (speaker, text) in enumerate(zip(speakers, utterances)):
                        if text:
                            emotion, confidence, _ = predict_emotion(text, model, tokenizer, config)
                            results.append({
                                'Speaker': speaker,
                                'Utterance': text,
                                'Emotion': emotion,
                                'Confidence': f"{confidence:.1%}"
                            })
                            emotions.append(emotion)
                
                # Display results
                st.subheader("Results")
                
                # Table
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Emotion flow
                st.subheader("Emotion Flow")
                flow_text = " → ".join([f"{get_emotion_emoji(e)} {e}" for e in emotions])
                st.markdown(f"### {flow_text}")
                
                # Summary
                col1, col2 = st.columns(2)
                with col1:
                    emotion_counts = pd.Series(emotions).value_counts()
                    fig = px.pie(
                        values=emotion_counts.values,
                        names=emotion_counts.index,
                        title="Emotion Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Dominant Emotion", emotion_counts.index[0].upper())
                    st.metric("Total Utterances", len(results))
                    st.metric("Unique Emotions", len(emotion_counts))
    
    # else:  # Batch Processing
    #     st.header("Batch Processing")
        
    #     uploaded_file = st.file_uploader(
    #         "Upload CSV file with text column:",
    #         type=['csv']
    #     )
        
    #     if uploaded_file:
    #         df = pd.read_csv(uploaded_file)
    #         st.write("Preview of uploaded file:")
    #         st.dataframe(df.head())
            
    #         # Column selection
    #         text_column = st.selectbox(
    #             "Select text column:",
    #             df.columns.tolist()
    #         )
            
    #         if st.button("🚀 Process All", type="primary"):
    #             progress_bar = st.progress(0)
    #             status_text = st.empty()
                
    #             results = []
    #             for i, text in enumerate(df[text_column]):
    #                 if pd.notna(text) and str(text).strip():
    #                     emotion, confidence, _ = predict_emotion(str(text), model, tokenizer, config)
    #                     results.append({
    #                         'text': text,
    #                         'emotion': emotion,
    #                         'confidence': confidence
    #                     })
                    
    #                 # Update progress
    #                 progress = (i + 1) / len(df)
    #                 progress_bar.progress(progress)
    #                 status_text.text(f"Processing... {i+1}/{len(df)}")
                
    #             # Add results to dataframe
    #             df['predicted_emotion'] = [r['emotion'] if i < len(results) else None for i, r in enumerate(results)]
    #             df['confidence'] = [r['confidence'] if i < len(results) else None for i, r in enumerate(results)]
                
    #             # Display results
    #             st.success("✅ Processing complete!")
    #             st.dataframe(df)
                
    #             # Download button
    #             csv = df.to_csv(index=False)
    #             st.download_button(
    #                 label="📥 Download Results",
    #                 data=csv,
    #                 file_name=f"emotion_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    #                 mime="text/csv"
    #             )
                
    #             # Visualizations
    #             col1, col2 = st.columns(2)
    #             with col1:
    #                 emotion_counts = df['predicted_emotion'].value_counts()
    #                 fig = px.bar(
    #                     x=emotion_counts.values,
    #                     y=emotion_counts.index,
    #                     orientation='h',
    #                     title="Emotion Distribution"
    #                 )
    #                 st.plotly_chart(fig, use_container_width=True)
                
    #             with col2:
    #                 avg_confidence = df.groupby('predicted_emotion')['confidence'].mean()
    #                 fig = px.bar(
    #                     x=avg_confidence.values,
    #                     y=avg_confidence.index,
    #                     orientation='h',
    #                     title="Average Confidence by Emotion"
    #                 )
    #                 st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()