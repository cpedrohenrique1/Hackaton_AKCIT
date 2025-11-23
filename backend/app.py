# app.py (arquivo completo; modo DEBUG EXTREMO)
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pymongo
from werkzeug.security import generate_password_hash, check_password_hash
from fpdf import FPDF
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
import os, uuid, traceback, sys, time

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
CORS(app)

# MongoDB
print("[INIT] Conectando ao MongoDB...")
client = pymongo.MongoClient(os.getenv("MONGO_URI"))



db = client["redacoesDB"]
usuarios = db["usuarios"]
print("[INIT] Mongo conectado. DB:", db.name, "Collection:", usuarios.name)

# Groq LLM (atenção: troque a chave se necessário)

from dotenv import load_dotenv
load_dotenv()

groq_key = os.getenv("CHAVEGROQ")
if not groq_key:
    raise ValueError("Variável CHAVEGROQ não encontrada no .env")

os.environ["GROQ_API_KEY"] = groq_key
print("[INIT] Configurando LLM (ChatGroq)...")
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)
print("[INIT] LLM criado:", llm)

# PDFs
os.makedirs("pdfs", exist_ok=True)
print("[INIT] Diretório 'pdfs' criado/verificado.")

# -----------------------------
# FUNÇÕES COM PROMPT AJUSTADO
# -----------------------------
def gerar_prompt_comp1(bloco: str) -> str:
    return f"""
Você é um avaliador profissional do ENEM especializado exclusivamente na Competência 1:
"Demonstra domínio da modalidade escrita formal da língua portuguesa".

Sua tarefa é **avaliar o BLOCO COMPLETO** fornecido, que pode conter de 1 a 5 parágrafos (tema, tese, argumentos e conclusão). 
Avalie todos os elementos textuais presentes, mesmo que estejam no meio do bloco.

⚠️ REGRAS IMPORTANTES (SIGA RIGOROSAMENTE):
- **NÃO RETORNE A REDAÇÃO**, nem trechos, nem resumos, nem paráfrases.
- **NÃO REESCREVA** o bloco, nem produza versão corrigida.
- O foco é **somente a análise**, nunca a reconstrução do texto.
- A resposta deve ser **limpa, objetiva, técnica e organizada**.
- Utilize **apenas linguagem avaliativa**, seguindo padrões usados por corretores do ENEM.

------------------------------------
CRITÉRIOS OFICIAIS QUE DEVEM SER ANALISADOS:
1. Estrutura sintática (períodos bem formados, articulação frasal, ausência de truncamentos).
2. Convenções da escrita (ortografia, acentuação, hífen, uso de maiúsculas e minúsculas).
3. Aspectos gramaticais (concordância, regência, tempos e modos verbais, pontuação, paralelismos).
4. Adequação ao registro formal exigido no ENEM (evitar informalidade e marcas de oralidade).
5. Vocabulário (precisão, adequação ao gênero dissertativo-argumentativo).

------------------------------------
TEXTO A SER AVALIADO (NÃO REPITA EM HIPÓTESE ALGUMA):
\"\"\"{bloco}\"\"\"

------------------------------------
FORMATO EXATO DA RESPOSTA:
1. Erros encontrados  
   Liste detalhadamente TODOS os problemas identificados no bloco.  
   Use numeração simples (1., 2., 3., ...).  
   Classifique cada erro brevemente (ex.: ortografia, sintaxe, concordância, registro, vocabulário etc.)

2. Recomendações objetivas de melhoria
   Indique COMO melhorar a escrita, de forma prática, técnica e direta.

3. Nota estimada (0 a 200 pontos)  
   Atribua uma nota seguindo estritamente a escala oficial:
   - 200 = domínio excelente; desvios mínimos ou inexistentes  
   - 160 = bom domínio; poucos desvios  
   - 120 = domínio mediano; erros perceptíveis  
   - 80  = domínio insuficiente; muitos desvios  
   - 40  = domínio precário; erros sistemáticos  
   - 0   = desconhecimento da modalidade formal
    """


# -----------------------------
# IMPORTS LANGCHAIN / FAISS / EMBEDDINGS
# -----------------------------
print("[INIT] Importando componentes FAISS / Embeddings...")
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# -----------------------------
# FERRAMENTA LANGCHAIN
# -----------------------------
@tool
def avaliar_paragrafo_comp1(bloco: str) -> str:
    """Ferramenta que gera o prompt de avaliação do bloco completo da redação."""
    return gerar_prompt_comp1(bloco)

print("[INIT] Criando agente LangChain com ferramenta de avaliação...")
agent = create_agent(
    model=llm,
    tools=[avaliar_paragrafo_comp1],
    system_prompt="Você é um corretor de redações do ENEM, competência 1."
)
print("[INIT] Agente criado:", agent)

def avaliar_paragrafo(bloco):
    # Debug: print do que será enviado
    print("\n[AGENT] --> Enviando ao agente (conteúdo do bloco):")
    print("---------- START BLOCO ----------")
    print(bloco)
    print("----------- END BLOCO -----------\n")

    try:
        # invoke e capturar exceções para debug
        resposta = agent.invoke({
            "messages": [{"role": "user", "content": bloco}]
        })
        print("[AGENT] <-- Resposta RAW do agente:")
        print(resposta)
        # extrair conteudo final
        final = resposta["messages"][-1].content if ("messages" in resposta and resposta["messages"]) else str(resposta)
        print("[AGENT] <-- Conteúdo extraído para retornar:")
        print(final)
        return final
    except Exception as e:
        print("[AGENT][ERROR] Exceção ao invocar agente:")
        traceback.print_exc()
        # retornar mensagem de erro amigável para front
        return f"[ERRO AGENTE] {repr(e)}\nVeja logs no servidor."

# -----------------------------
# FUNÇÃO PARA GERAR PDF (UTF-8 safe)
# -----------------------------
def gerar_pdf(email, bloco, avaliacao):
    print("[PDF] Gerando PDF...")
    pdf = FPDF()
    pdf.add_page()

    # Tentar registrar DejaVu (suporta unicode) - se não existir, fallback
    font_added = False
    try:
        font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
        print(f"[PDF] Tentando carregar fonte DejaVu em: {font_path}")
        if os.path.isfile(font_path):
            pdf.add_font("DejaVu", "", font_path, uni=True)
            pdf.set_font("DejaVu", size=12)
            font_added = True
            print("[PDF] Fonte DejaVu carregada com sucesso.")
        else:
            raise FileNotFoundError("arquivo TTF não encontrado")
    except Exception as e:
        print("[PDF][WARN] Não foi possível carregar DejaVu (fallback para latin-1). Erro:", e)
        try:
            pdf.set_font("Arial", size=12)
            print("[PDF] Fonte Arial selecionada (padrão).")
        except Exception as e2:
            print("[PDF][WARN] Arial também não disponível:", e2)
            # irá falhar se nenhuma fonte, mas seguimos

    conteudo = f"Email: {email}\n\nRedação (bloco):\n{bloco}\n\nAvaliação:\n{avaliacao}"

    # Se DejaVu não foi adicionada, converter caracteres não-latin1 para '?'
    if not font_added:
        try:
            conteudo = conteudo.encode('latin-1', errors='replace').decode('latin-1')
            print("[PDF] Aplicado encode('latin-1', errors='replace') ao conteúdo do PDF para evitar Unicode errors.")
        except Exception as e:
            print("[PDF][ERROR] Falha ao converter conteúdo para latin-1:", e)

    # Debug comprimento do conteúdo e primeiros 300 chars
    print(f"[PDF] Conteúdo tamanho: {len(conteudo)} chars. Preview (300 chars):\n{conteudo[:300]}")

    pdf.multi_cell(0, 8, conteudo)

    nome_pdf = f"avaliacao_{uuid.uuid4()}.pdf"
    caminho = os.path.join("pdfs", nome_pdf)
    try:
        pdf.output(caminho)
        print("[PDF] PDF salvo em:", caminho)
    except Exception as e:
        print("[PDF][ERROR] Falha ao salvar PDF:", e)
        traceback.print_exc()
        # tentar salvar de forma simplificada
        try:
            with open(caminho, "wb") as f:
                f.write(b"")  # fallback: criar arquivo vazio
            print("[PDF] Criado arquivo vazio como fallback:", caminho)
        except Exception as e2:
            print("[PDF][ERROR] Falha no fallback de gravação do PDF:", e2)
    return nome_pdf

# -----------------------------
# FAISS: inicializar ANTES do servidor rodar
# -----------------------------
print("\n[FAISS] Inicializando FAISS e embeddings...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
print("[FAISS] Embeddings instanciados:", embeddings_model)

# Documento base para evitar erro de lista vazia
initial_title = "Documento Inicial - Base FAISS"
initial_text = (
    "Texto base para iniciar o índice FAISS. "
    "Usado apenas para evitar erros na inicialização e permitir add_documents depois."
)
initial_doc = Document(
    page_content=f"Título: {initial_title}\n\nRedação:\n{initial_text}",
    metadata={"titulo": initial_title, "source": "seed"}
)

print("[FAISS] Criando índice FAISS.from_documents com 1 doc seed...")
faiss_index = FAISS.from_documents(
    documents=[initial_doc],
    embedding=embeddings_model
)
print("[FAISS] FAISS inicializado com seed. index object:", faiss_index)

# debug: print estrutura básica (não prints de embeddings pesados)
print("[FAISS] Ready. Você pode adicionar documentos com faiss_index.add_documents([...])\n")

# -----------------------------
# Função util: criar chunk (5 blocos)
# -----------------------------
def criar_chunk_completo(paragrafos, titulo):
    print("[CHUNK] Criando chunk com até 5 blocos a partir dos parágrafos recebidos...")
    print("[CHUNK] Parágrafos originais count:", len(paragrafos))
    # limpar parágrafos que são apenas whitespace
    cleaned = [p.strip() for p in paragrafos if p is not None]
    print("[CHUNK] Parágrafos limpos (trim):", cleaned)
    while len(cleaned) < 5:
        cleaned.append("")
    # junta os primeiros 5 (mesmo que vazios)
    bloco = "\n\n".join(cleaned[:5])
    chunk = f"Título: {titulo}\n\nRedação:\n{bloco}"
    print("[CHUNK] Chunk criado (preview 500 chars):")
    print(chunk[:500])
    return chunk

# -----------------------------
# ROTA PRINCIPAL (DEBUG EXTREMO)
# -----------------------------
@app.route("/enviar_redacao", methods=["POST"])
def enviar_redacao():
    print("\n" + "="*80)
    print("[ROTA] /enviar_redacao chamada")
    print("="*80)

    start_time = time.time()
    data = request.json or {}
    print("[ROTA] Payload JSON recebido:", data.keys())

    email = data.get("email")
    redacao = data.get("redacao")
    titulo = data.get("titulo")

    print("[ROTA] email:", email)
    print("[ROTA] titulo:", titulo)
    print("[ROTA] redacao (preview 400 chars):")
    print((redacao or "")[:400])

    if not email:
        print("[ROTA][ERROR] email ausente no payload.")
        return jsonify({"erro": "Email ausente."}), 400
    if not titulo:
        print("[ROTA][ERROR] titulo ausente no payload.")
        return jsonify({"erro": "Título não informado."}), 400
    if not redacao:
        print("[ROTA][ERROR] redacao ausente no payload.")
        return jsonify({"erro": "Redação vazia."}), 400

    # separar parágrafos por linhas vazias (mantendo blocos)
    # estratégia simples: split por linha em branco dupla ou única quebra
    paragrafos = [p for p in [seg.strip() for seg in redacao.split("\n")] if p != ""]
    # se user realmente quer tratar cada newline como parágrafo, substitua a linha acima por:
    # paragrafos = redacao.split("\n")
    print("[ROTA] Parágrafos detectados (count):", len(paragrafos))
    for i, p in enumerate(paragrafos):
        print(f"[ROTA] Parágrafo {i+1} (len {len(p)}): {p[:200]}")

    # criar chunk com até 5 blocos
    chunk_completo = criar_chunk_completo(paragrafos, titulo)

    # criar Document e adicionar ao FAISS
    doc_id = str(uuid.uuid4())
    novo_doc = Document(
        page_content=chunk_completo,
        metadata={"titulo": titulo, "usuario": email, "id_redacao": doc_id}
    )
    print("[FAISS] Document a ser adicionado (id):", doc_id)
    print("[FAISS] page_content preview (300 chars):")
    print(novo_doc.page_content[:300])

    try:
        faiss_index.add_documents([novo_doc])
        print("[FAISS] Documento adicionado com sucesso.")
    except Exception as e:
        print("[FAISS][ERROR] Falha ao adicionar documento:")
        traceback.print_exc()
        return jsonify({"erro": "Falha ao adicionar documento ao FAISS", "detail": str(e)}), 500

    # busca similar
    try:
        print("[FAISS] Executando similarity_search com o chunk_completo ...")
        resultados = faiss_index.similarity_search(chunk_completo, k=1)
        print("[FAISS] Resultados obtidos (len):", len(resultados))
        for idx, r in enumerate(resultados):
            print(f"[FAISS] Resultado {idx} preview (300 chars):")
            print(r.page_content[:300])
    except Exception as e:
        print("[FAISS][ERROR] Falha na busca similarity_search:")
        traceback.print_exc()
        resultados = []

    chunk_para_avaliacao = resultados[0].page_content if resultados else chunk_completo
    print("[ROTA] Chunk que será enviado para avaliação (preview 500 chars):")
    print(chunk_para_avaliacao[:500])

    # enviar ao agente e capturar resposta
    try:
        print("[ROTA] Chamando função avaliar_paragrafo(...)")
        avaliacao = avaliar_paragrafo(chunk_para_avaliacao)
        print("[ROTA] Avaliação recebida (preview 500 chars):")
        print((avaliacao or "")[:500])
    except Exception as e:
        print("[ROTA][ERROR] Erro ao avaliar parágrafo com o agente:")
        traceback.print_exc()
        avaliacao = f"[ERRO AVALIAÇÃO] {repr(e)}"

    # gerar pdf
    try:
        pdf_nome = gerar_pdf(email, chunk_para_avaliacao, avaliacao)
    except Exception as e:
        print("[ROTA][ERROR] Erro ao gerar PDF:")
        traceback.print_exc()
        pdf_nome = None

    # salvar no MongoDB
    try:
        user = usuarios.find_one({"email": email})
        print("[MONGO] usuário buscado:", user)
        if not user:
            # se usuário não existe, criamos um com redacoes vazias
            print("[MONGO] Usuário não existe — criando registro novo.")
            usuarios.insert_one({"email": email, "senha": generate_password_hash("temporal"), "redacoes": []})
            user = usuarios.find_one({"email": email})
            print("[MONGO] Usuário criado:", user)

        historico = user.get("redacoes", []) or []
        nova_redacao = {
            "id": doc_id,
            "titulo": titulo,
            "paragrafos": paragrafos,
            "avaliacao": avaliacao,
            "pdf": pdf_nome,
            "timestamp": time.time()
        }
        historico.append(nova_redacao)
        print("[MONGO] Atualizando histórico com nova redação:", nova_redacao)
        usuarios.update_one({"email": email}, {"$set": {"redacoes": historico}})
        print("[MONGO] update_one concluído.")
    except Exception as e:
        print("[MONGO][ERROR] Erro ao salvar no MongoDB:")
        traceback.print_exc()
        return jsonify({"erro": "Falha ao salvar no MongoDB", "detail": str(e)}), 500

    end_time = time.time()
    duration = round(end_time - start_time, 3)
    print(f"[ROTA] /enviar_redacao finalizada em {duration} segundos")

    return jsonify({
        "avaliacao": avaliacao,
        "pdf_url": f"/pdfs/{pdf_nome}" if pdf_nome else None,
        "titulo": titulo,
        "doc_id": doc_id,
        "debug": {
            "faiss_results_count": len(resultados) if 'resultados' in locals() else 0,
            "duration_s": duration
        }
    })

# -----------------------------
# ROTAS AUXILIARES (registro/login/historico/pdf)
# -----------------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.json or {}
    email = data.get("email")
    senha = data.get("senha")
    print(f"[ROTA] /register chamado com {email}")
    if not email or not senha:
        return jsonify({"erro": "E-mail e senha são obrigatórios."}), 400
    if usuarios.find_one({"email": email}):
        return jsonify({"erro": "Email já registrado."}), 400
    usuarios.insert_one({"email": email, "senha": generate_password_hash(senha), "redacoes": []})
    print("[ROTA] usuário registrado:", email)
    return jsonify({"mensagem": "Registrado com sucesso."})

@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    email = data.get("email")
    senha = data.get("senha")
    print(f"[ROTA] /login chamado com {email}")
    usuario = usuarios.find_one({"email": email})
    if usuario and check_password_hash(usuario["senha"], senha):
        print("[ROTA] login bem sucedido")
        return jsonify({"mensagem": "Login bem-sucedido."})
    print("[ROTA] credenciais inválidas")
    return jsonify({"erro": "Credenciais inválidas."}), 401

@app.route("/historico", methods=["GET"])
def historico():
    email = request.args.get("email")
    titulo = request.args.get("titulo")
    print(f"[ROTA] /historico chamado. email={email} titulo={titulo}")
    user = usuarios.find_one({"email": email})
    if not user:
        return jsonify({"erro": "Usuário não encontrado."}), 404
    redacoes = user.get("redacoes", []) or []
    if titulo:
        redacoes = [r for r in redacoes if titulo.lower() in r.get("titulo","").lower()]
    print("[ROTA] histórico retornado count:", len(redacoes))
    return jsonify({"redacoes": redacoes})

@app.route("/pdfs/<nome_pdf>")
def servir_pdf(nome_pdf):
    print(f"[ROTA] Servindo PDF: {nome_pdf}")
    return send_from_directory("pdfs", nome_pdf)

@app.route("/")
def index():
    return send_from_directory(frontend_dir, "registro.html")

@app.route("/<path:filename>")
def serve_front(filename):
    return send_from_directory(frontend_dir, filename)

# -----------------------------
# START
# -----------------------------
if __name__ == "__main__":
    print("[MAIN] Iniciando servidor Flask (debug ON)...")
    os.makedirs("pdfs", exist_ok=True)
    app.run(debug=True)
