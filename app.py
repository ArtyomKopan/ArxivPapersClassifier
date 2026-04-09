import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import numpy as np
from pathlib import Path
import json

st.set_page_config(
    page_title='arXiv Papers Classifier',
    page_icon='📚',
    layout='wide'
)

@st.cache_resource
def load_model_and_tokenizer(model_path='my_model'):
    device = 'cpu'
    
    try:
        model_dir = Path(model_path)
        config = AutoConfig.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            config=config
        )
        model.to(device)
        model.eval()
        
        label_mapping_path = model_dir / 'label_mapping.json'
        id_to_label = None
        
        if label_mapping_path.exists():
            with open(label_mapping_path, 'r') as f:
                label_mapping = json.load(f)
                id_to_label = {int(k): v for k, v in label_mapping['id_to_label'].items()}
        
        return model, tokenizer, device, id_to_label
    
    except Exception as e:
        st.error(f'Ошибка загрузки модели: {str(e)}')
        return None, None, None, None

def get_top_categories_cumulative(probabilities, id_to_label, threshold=0.95):
    sorted_indices = np.argsort(probabilities)[::-1]
    
    cumulative_prob = 0.0
    selected_categories = []
    
    for idx in sorted_indices:
        prob = probabilities[idx]
        cumulative_prob += prob
        category = id_to_label[idx] if id_to_label else f'Class_{idx}'
        selected_categories.append(category)
        
        if cumulative_prob >= threshold:
            break
    
    return selected_categories

def predict_category(title, abstract, model, tokenizer, device, id_to_label):
    text = f'{title} [SEP] {abstract}' if abstract.strip() else text = title
    
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
    
    probs = probabilities.cpu().numpy()[0]
    
    selected_categories = get_top_categories_cumulative(probs, id_to_label, threshold=0.95)
    
    return selected_categories

def main():
    st.title('arXiv papers classifier')
    
    with st.spinner('Loading model and tokenizer...'):
        model, tokenizer, device, id_to_label = load_model_and_tokenizer()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        title = st.text_area(
            '**Input title:**',
            placeholder='Enter the paper title here...',
            height=100,
            key='title_input'
        )
        
        abstract = st.text_area(
            '**Input abstract:**',
            placeholder='Enter the abstract here...',
            height=300,
            key='abstract_input'
        )
        
        classify_button = st.button('Classify Paper', type='primary', use_container_width=True)
    
    with col2:
        st.markdown('**The most probable categories**')
        st.caption('(showing until cumulative probability ≥ 95%)')
        
        results_container = st.container(border=True)
        
        with results_container:
            if classify_button:
                if not title.strip():
                    st.warning('Please enter at least a paper title.')
                else:
                    progress_bar = st.progress(0, text='Classifying...')
                    
                    try:
                        progress_bar.progress(50, text='Processing text...')
                        
                        selected_categories = predict_category(
                            title, abstract, model, tokenizer, 
                            device, id_to_label
                        )
                        
                        progress_bar.progress(100, text='Complete!')
                        
                        for i, category in enumerate(selected_categories, 1):
                            st.markdown(f'**{i}.** `{category}`')
                                            
                    except Exception as e:
                        st.error(f'Ошибка при классификации: {str(e)}')
                    
                    finally:
                        progress_bar.empty()
            
            else:
                st.info('Enter title and abstract and click on the button to see results')

if __name__ == '__main__':
    main()
