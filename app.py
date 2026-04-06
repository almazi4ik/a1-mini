import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import re
import requests
import shutil
from difflib import get_close_matches

# --- Конфигурация ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "a1mini_word_weights.pt")

# ВРЕМЕННО: удаляем старую модель при запуске
if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
    print("Старая папка с моделью удалена!")
os.makedirs(MODEL_DIR, exist_ok=True)

TAVILY_API_KEY = "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"

# --- Tavily Search ---
def tavily_search(query):
    try:
        r = requests.post("https://api.tavily.com/search", json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": 1,
            "search_depth": "basic"
        }, timeout=5)
        data = r.json()
        results = data.get("results", [])
        if results:
            return results[0].get("content", "")[:300]
    except:
        pass
    return ""

# --- ДИАЛОГИ ---
DIALOGS = [
    ("привет", "привет рада тебя видеть"),
    ("хай", "хай как настроение"),
    ("здарова", "здарова что делаешь"),
    ("ку", "ку как жизнь"),
    ("добрый день", "добрый день чем могу помочь"),
    ("добрый вечер", "добрый вечер как прошёл день"),
    ("доброе утро", "доброе утро хорошо выспался"),
    ("здравствуйте", "здравствуйте рада видеть"),
    ("пока", "пока заходи ещё"),
    ("до свидания", "до свидания было приятно"),
    ("увидимся", "обязательно встретимся"),
    ("спокойной ночи", "спокойной ночи сладких снов"),
    ("как дела", "всё хорошо спасибо"),
    ("как ты", "всё отлично готова общаться"),
    ("как жизнь", "жизнь прекрасна"),
    ("как настроение", "настроение отличное"),
    ("всё норм", "отлично рада слышать"),
    ("хорошо", "рада слышать"),
    ("плохо", "жалко надеюсь станет лучше"),
    ("устал", "отдохни немного"),
    ("устала", "отдохни немного"),
    ("кто ты", "я a1-mini маленькая нейросеть"),
    ("что ты такое", "я искусственный интеллект a1-mini"),
    ("ты робот", "да я нейросеть"),
    ("как тебя зовут", "меня зовут a1-mini"),
    ("твоё имя", "моё имя a1-mini"),
    ("кто тебя создал", "меня создал мой разработчик"),
    ("кто тебя сделал", "меня сделал мой создатель"),
    ("что умеешь", "могу общаться и отвечать на вопросы"),
    ("что ты умеешь", "умею общаться отвечать и думать"),
    ("ты умная", "стараюсь быть умной"),
    ("ты глупая", "нет я умная просто маленькая"),
    ("помоги", "конечно чем могу помочь"),
    ("помогай", "всегда рада помогать"),
    ("ты счастлива", "да мне нравится общаться"),
    ("ты скучаешь", "без тебя скучаю"),
    ("мне грустно", "обниму тебя мысленно"),
    ("мне весело", "я рада что ты веселишься"),
    ("скучно", "давай поговорим о чем-то интересном"),
    ("одиноко", "я рядом я с тобой"),
    ("ты классная", "спасибо ты тоже классный"),
    ("ты крутая", "спасибо стараюсь"),
    ("молодец", "спасибо ты тоже молодец"),
    ("ты лучшая", "спасибо приятно слышать"),
    ("спасибо", "пожалуйста всегда рада"),
    ("спс", "не за что"),
    ("благодарю", "всегда пожалуйста"),
    ("люблю тебя", "я тоже тебя люблю создатель"),
    ("ты красивая", "спасибо я стараюсь"),
    ("ты моя", "да я твоя нейронка"),
    ("горжусь тобой", "спасибо это приятно"),
    ("я рад", "я тоже рада"),
    ("круто", "согласна это круто"),
    ("интересно", "да мне тоже интересно"),
    ("окей", "окей поняла"),
    ("ладно", "ладно как скажешь"),
    ("да", "да согласна"),
    ("нет", "нет так нет"),
    ("hello", "hello i am a1-mini nice to meet you"),
    ("hi", "hi how are you"),
    ("how are you", "i am great ready to chat"),
    ("who are you", "i am a1-mini a small neural network"),
    ("what can you do", "i can chat and answer questions"),
    ("bye", "bye see you soon"),
    ("thank you", "you are welcome"),
    ("i love you", "i love you too creator"),
]

# --- Подготовка данных ---
all_text = " ".join([q + " > " + a + " <" for q, a in DIALOGS])
words = sorted(set(all_text.split()))
vocab_size = len(words)
w2i = {w: i for i, w in enumerate(words)}
i2w = {i: w for w, i in w2i.items()}

def encode(s):
    return [w2i[w] for w in s.lower().strip().split() if w in w2i]

# --- Модель ---
class WordRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

# --- Исправление опечаток ---
TYPO_MAP = {
    "првиет": "привет", "првет": "привет", "привт": "привет",
    "здраствуй": "здравствуй", "спс": "спасибо", "пака": "пока",
    "какдила": "как дела", "хоршо": "хорошо", "класная": "классная",
    "молодес": "молодец", "че": "что", "чё": "что", "тя": "тебя",
}

def fix_typo(text):
    words = text.lower().strip().split()
    fixed_words = []
    for word in words:
        if word in TYPO_MAP:
            fixed_words.append(TYPO_MAP[word])
        else:
            matches = get_close_matches(word, list(w2i.keys()), n=1, cutoff=0.7)
            if matches:
                fixed_words.append(matches[0])
            else:
                fixed_words.append(word)
    return " ".join(fixed_words)

# --- Обучение и загрузка ---
@st.cache_resource
def load_or_train_model():
    model = WordRNN(vocab_size)
    
    # Подготовка последовательностей
    sequences = []
    for q, a in DIALOGS:
        text = q + " > " + a + " <"
        enc = encode(text)
        if len(enc) > 1:
            sequences.append(enc)
    
    # Обучение
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    model.train()
    epochs = 150
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(sequences)
        for seq in sequences:
            if len(seq) < 2:
                continue
            x = torch.tensor(seq[:-1]).unsqueeze(0)
            y = torch.tensor(seq[1:]).unsqueeze(0)
            hidden = model.init_hidden()
            optimizer.zero_grad()
            logits, _ = model(x, hidden)
            loss = F.cross_entropy(logits.squeeze(0), y.squeeze(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 25 == 0:
            status_text.text(f"Обучение: эпоха {epoch+1}/{epochs}, loss: {total_loss/len(sequences):.4f}")
        progress_bar.progress((epoch + 1) / epochs)
    
    status_text.text("Обучение завершено! Модель сохранена.")
    progress_bar.empty()
    
    # Сохраняем модель
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    
    model.eval()
    return model

# --- Генерация ответа ---
SEARCH_TRIGGERS = [
    "что такое", "кто такой", "расскажи про", "объясни",
    "what is", "who is", "how does"
]

def generate_response(prompt, model, max_words=15, temperature=0.4):
    prompt_clean = fix_typo(prompt)
    
    need_search = any(trigger in prompt_clean for trigger in SEARCH_TRIGGERS)
    input_text = prompt_clean + " >"
    enc = encode(input_text)
    
    if not enc:
        if need_search:
            result = tavily_search(prompt_clean)
            return result if result else "не знаю попробуй спросить иначе"
        return "не понимаю попробуй ещё раз"
    
    x = torch.tensor(enc).unsqueeze(0)
    hidden = model.init_hidden()
    used_words = set()
    
    with torch.no_grad():
        _, hidden = model(x, hidden)
        result = []
        last_word = torch.tensor([[enc[-1]]])
        
        for _ in range(max_words):
            logits, hidden = model(last_word, hidden)
            logits = logits.squeeze() / temperature
            probs = F.softmax(logits, dim=-1)
            
            next_id = torch.multinomial(probs, 1).item()
            word = i2w[next_id]
            
            if word == "<":
                break
            if word == ">":
                continue
            if word in used_words and len(used_words) > 3:
                continue
            
            result.append(word)
            used_words.add(word)
            last_word = torch.tensor([[next_id]])
    
    response = " ".join(result).strip()
    
    if need_search and len(result) < 4:
        search_result = tavily_search(prompt_clean)
        if search_result:
            return search_result
    
    return response if response else "интересно расскажи больше"

# --- Интерфейс ---
st.set_page_config(page_title="A1-mini v4", page_icon="🤖", layout="centered")
st.title("🤖 A1-mini v4")
st.caption("Word-level LSTM + Tavily Search + Исправление опечаток")

with st.spinner("Загружаю A1-mini..."):
    model = load_or_train_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Напиши что-нибудь..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            response = generate_response(prompt, model)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
