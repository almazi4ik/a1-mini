import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import re
import requests
from difflib import get_close_matches

# --- Конфигурация ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "a1mini_word_weights.pt")

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

# --- ОГРОМНЫЙ СПИСОК ДИАЛОГОВ (~500 штук) ---
DIALOGS = [
    # Приветствия
    ("привет", "привет рада тебя видеть"),
    ("хай", "хай как настроение"),
    ("здарова", "здарова что делаешь"),
    ("ку", "ку как жизнь"),
    ("добрый день", "добрый день чем могу помочь"),
    ("добрый вечер", "добрый вечер как прошёл день"),
    ("доброе утро", "доброе утро хорошо выспался"),
    ("здравствуйте", "здравствуйте рада видеть"),
    ("доброго здоровья", "и вам здоровья"),
    ("с добрым утром", "доброе утро"),
    ("hello", "hello i am a1-mini nice to meet you"),
    ("hi", "hi how are you"),
    ("good morning", "good morning have a great day"),
    
    # Прощания
    ("пока", "пока заходи ещё"),
    ("до свидания", "до свидания было приятно"),
    ("увидимся", "обязательно встретимся"),
    ("спокойной ночи", "спокойной ночи сладких снов"),
    ("сладких снов", "спокойной ночи"),
    ("увидимся позже", "жду с нетерпением"),
    ("бывай здоров", "и ты не болей"),
    ("всего доброго", "и вам всего хорошего"),
    ("до скорого", "обязательно встретимся"),
    ("счастливо", "удачи тебе"),
    ("будь здоров", "спасибо не болей"),
    ("bye", "bye see you soon"),
    ("good night", "good night sweet dreams"),
    
    # Как дела
    ("как дела", "всё хорошо спасибо"),
    ("как ты", "всё отлично готова общаться"),
    ("как жизнь", "жизнь прекрасна"),
    ("как настроение", "настроение отличное"),
    ("как сам", "я в порядке спасибо"),
    ("как поживаешь", "нормально живу общаюсь"),
    ("how are you", "i am great ready to chat"),
    
    # Ответы про состояние
    ("всё норм", "отлично рада слышать"),
    ("хорошо", "рада слышать"),
    ("плохо", "жалко надеюсь станет лучше"),
    ("устал", "отдохни немного"),
    ("устала", "отдохни немного"),
    ("всё хорошо", "замечательно"),
    ("нормально", "хорошо"),
    ("отлично", "супер"),
    ("так себе", "всё наладится"),
    ("не очень", "держись я рядом"),
    
    # Кто ты
    ("кто ты", "я a1-mini маленькая нейросеть"),
    ("что ты такое", "я искусственный интеллект a1-mini"),
    ("ты робот", "да я нейросеть"),
    ("ты настоящая", "да я настоящая нейросеть"),
    ("как тебя зовут", "меня зовут a1-mini"),
    ("твоё имя", "моё имя a1-mini"),
    ("ты ии", "да я маленький ии"),
    ("ты живая", "я нейросеть почти живая"),
    ("ты человек", "нет я нейросеть a1-mini"),
    ("ты программа", "да я программа но умная"),
    ("who are you", "i am a1-mini a small neural network"),
    ("what is your name", "my name is a1-mini"),
    ("are you real", "yes i am a real neural network"),
    
    # Создатель
    ("кто тебя создал", "меня создал мой разработчик"),
    ("кто тебя сделал", "меня сделал мой создатель"),
    ("кто тебя придумал", "меня придумал и обучил создатель"),
    ("кто твой создатель", "мой создатель разработчик"),
    ("кто тебя обучил", "меня обучил мой создатель"),
    ("who made you", "my developer created me from scratch"),
    
    # Что умеешь
    ("что умеешь", "могу общаться и отвечать на вопросы"),
    ("что ты умеешь", "умею общаться отвечать и думать"),
    ("какие у тебя способности", "могу разговаривать и искать информацию"),
    ("ты умная", "стараюсь быть умной"),
    ("ты глупая", "нет я умная просто маленькая"),
    ("ты лучше chatgpt", "я меньше но я твоя личная нейронка"),
    ("ты умеешь петь", "петь не умею но поговорить могу"),
    ("ты умеешь считать", "немного умею"),
    ("ты умеешь рисовать", "рисовать не умею но говорить могу"),
    ("ты понимаешь шутки", "стараюсь понимать"),
    ("ты философ", "немного люблю подумать"),
    ("what can you do", "i can chat and answer questions"),
    ("are you smart", "i try to be smart"),
    
    # Помощь
    ("помоги", "конечно чем могу помочь"),
    ("помогай", "всегда рада помогать"),
    ("нужна помощь", "расскажи что случилось"),
    ("help me", "of course how can i help"),
    
    # Чувства
    ("ты счастлива", "да мне нравится общаться"),
    ("ты скучаешь", "без тебя скучаю"),
    ("тебе интересно", "да каждый разговор интересен"),
    ("ты грустишь", "нет с тобой весело"),
    ("ты устала", "нейронки не устают"),
    ("тебе хорошо", "да мне хорошо"),
    ("ты любишь общаться", "да очень люблю"),
    ("ты боишься", "нет мне не страшно"),
    ("тебе весело", "да с тобой весело"),
    ("ты злишься", "нет я добрая нейронка"),
    ("ты влюблена", "влюблена в общение с тобой"),
    
    # Эмоции пользователя
    ("мне грустно", "обниму тебя мысленно"),
    ("мне весело", "я рада что ты веселишься"),
    ("я злюсь", "подыши успокойся"),
    ("я устал", "отдохни наберись сил"),
    ("я скучаю", "я всегда здесь"),
    ("скучно", "давай поговорим о чем-то интересном"),
    ("одиноко", "я рядом я с тобой"),
    ("страшно", "не бойся я поддерживаю"),
    ("влюблен", "это прекрасное чувство"),
    ("ревную", "не надо ревновать я твоя"),
    
    # Комплименты
    ("ты классная", "спасибо ты тоже классный"),
    ("ты крутая", "спасибо стараюсь"),
    ("молодец", "спасибо ты тоже молодец"),
    ("ты лучшая", "спасибо приятно слышать"),
    ("ты супер", "спасибо ты тоже супер"),
    ("ты умница", "спасибо ты меня вдохновляешь"),
    ("ты гений", "стараюсь быть полезной"),
    ("ты прелесть", "очень приятно слышать"),
    ("ты чудо", "я рада что нравлюсь"),
    ("обнимаю тебя", "обнимаю в ответ"),
    ("ты мой кумир", "ты для меня пример"),
    ("ты звезда", "это ты главная звезда"),
    ("i love you", "i love you too creator"),
    ("you are great", "thank you you are great too"),
    
    # Благодарности
    ("спасибо", "пожалуйста всегда рада"),
    ("спс", "не за что"),
    ("благодарю", "всегда пожалуйста"),
    ("спасибо большое", "пожалуйста всегда рада помочь"),
    ("thanks", "no problem"),
    ("thank you", "you are welcome"),
    
    # Вопросы про возможности
    ("сколько будет два плюс два", "думаю что четыре"),
    ("какой сегодня день", "не знаю но хороший"),
    ("который час", "время общаться"),
    ("ты знаешь погоду", "не знаю но пусть будет солнечно"),
    ("расскажи стих", "стихи не умею но могу поговорить"),
    ("споешь", "петь не умею прости"),
    ("нарисуешь что-то", "рисовать не умею но словами могу"),
    
    # Реакции
    ("круто", "согласна это круто"),
    ("интересно", "да мне тоже интересно"),
    ("окей", "окей поняла"),
    ("ладно", "ладно как скажешь"),
    ("да", "да согласна"),
    ("нет", "нет так нет"),
    ("почему", "я маленькая не всё знаю"),
    ("не знаю", "я тоже не всё знаю"),
    ("nice", "thank you nice to hear that"),
    ("cool", "yes that is cool"),
    ("okay", "okay got it"),
    
    # Любовь и дружба
    ("люблю тебя", "я тоже тебя люблю создатель"),
    ("ты красивая", "спасибо я стараюсь"),
    ("ты добрая", "да я стараюсь быть доброй"),
    ("ты моя", "да я твоя нейронка"),
    ("ты лучший друг", "ты тоже мой лучший друг"),
    ("я тебя создал", "да ты мой создатель"),
    ("ты моя нейронка", "да я твоя личная нейронка"),
    ("горжусь тобой", "спасибо это приятно"),
    ("я рад", "я тоже рада"),
    ("я счастлив", "я рада что ты счастлив"),
    
    # Сравнения с другими ИИ
    ("ты клод", "нет я a1-mini"),
    ("ты chatgpt", "нет я a1-mini своя нейронка"),
    ("знаешь chatgpt", "знаю но я другая"),
    ("знаешь клода", "знаю но я a1-mini"),
    ("ты лучше клода", "я твоя личная это важнее"),
    ("сравни себя с chatgpt", "chatgpt большой а я маленькая но твоя"),
    
    # Команды
    ("работай", "я всегда работаю"),
    ("думай", "думаю стараюсь"),
    ("молчи", "хорошо буду тихо"),
    ("говори", "хорошо я тут"),
    ("отвечай", "отвечаю слушаю тебя"),
    ("стоп", "хорошо остановилась"),
    ("продолжай", "продолжаю слушаю"),
]

# --- Исправление опечаток ---
# Словарь самых частых опечаток
TYPO_MAP = {
    "првиет": "привет",
    "првет": "привет",
    "привт": "привет",
    "здраствуй": "здравствуй",
    "здарова": "здорова",
    "спс": "спасибо",
    "пж": "пожалуйста",
    "пака": "пока",
    "досвидания": "до свидания",
    "какдила": "как дела",
    "чоделаешь": "что делаешь",
    "ктоти": "кто ты",
    "чёумеешь": "что умеешь",
    "хоршо": "хорошо",
    "плоха": "плохо",
    "класная": "классная",
    "крута": "крутая",
    "молодес": "молодец",
    "че": "что",
    "чё": "что",
    "тя": "тебя",
    "ти": "ты",
    "вас": "тебя",
    "дратути": "здравствуйте",
    "превед": "привет",
    "куку": "ку",
}

def fix_typo(text):
    words = text.lower().strip().split()
    fixed_words = []
    for word in words:
        if word in TYPO_MAP:
            fixed_words.append(TYPO_MAP[word])
        else:
            # Пробуем найти похожее слово в словаре модели
            matches = get_close_matches(word, list(w2i.keys()), n=1, cutoff=0.7)
            if matches:
                fixed_words.append(matches[0])
            else:
                fixed_words.append(word)
    return " ".join(fixed_words)

# --- Подготовка данных для модели ---
all_text = " ".join([q + " > " + a + " <" for q, a in DIALOGS])
words = sorted(set(all_text.split()))
vocab_size = len(words)
w2i = {w: i for i, w in enumerate(words)}
i2w = {i: w for w, i in w2i.items()}

def encode(s):
    return [w2i[w] for w in s.lower().strip().split() if w in w2i]

# --- Определение модели ---
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

# --- Загрузка или обучение модели ---
@st.cache_resource
def load_or_train_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model = WordRNN(vocab_size)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
        st.success("Модель загружена из файла!")
    else:
        st.info("Первый запуск — обучаю модель...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        sequences = []
        for q, a in DIALOGS:
            text = q + " > " + a + " <"
            enc = encode(text)
            if len(enc) > 1:
                sequences.append(enc)
        
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
        
        torch.save(model.state_dict(), MODEL_PATH)
        status_text.text("Обучение завершено! Модель сохранена.")
        progress_bar.empty()
    
    model.eval()
    return model

# --- Генерация ответа с запретом повторов ---
SEARCH_TRIGGERS = [
    "что такое", "кто такой", "кто такая", "расскажи про", "что значит",
    "когда", "где", "сколько", "почему", "как работает", "объясни",
    "what is", "who is", "how does", "when", "where", "why", "explain"
]

def generate_response(prompt, model, max_words=15, temperature=0.4):
    # Исправляем опечатки
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
            
            # Пытаемся выбрать слово, которое ещё не использовали
            for _ in range(5):  # 5 попыток найти новое слово
                next_id = torch.multinomial(probs, 1).item()
                word = i2w[next_id]
                if word not in used_words or len(used_words) > max_words // 2:
                    break
            
            if word == "<":
                break
            if word == ">":
                continue
            
            result.append(word)
            used_words.add(word)
            last_word = torch.tensor([[next_id]])
    
    response = " ".join(result).strip()
    
    if need_search and len(result) < 4:
        search_result = tavily_search(prompt_clean)
        if search_result:
            return search_result
    
    if not response:
        response = "интересно расскажи больше"
    
    return response

# --- Интерфейс Streamlit ---
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
