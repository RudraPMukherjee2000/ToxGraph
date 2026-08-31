import streamlit as st
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
from supabase import create_client
from atproto import Client as BskyClient
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="ToxGraph | Focal Structure Analysis", layout="wide")

# --- STATE MANAGEMENT ---
if 'node_data' not in st.session_state:
    st.session_state.node_data = {}
if 'graph_nodes' not in st.session_state:
    st.session_state.graph_nodes = []
if 'graph_edges' not in st.session_state:
    st.session_state.graph_edges = []
if 'has_data' not in st.session_state:
    st.session_state.has_data = False

# --- 1. CACHING DATABASE & CLIENT CONNECTIONS ---
@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

@st.cache_resource
def init_bsky_client():
    client = BskyClient()
    client.login(st.secrets["BLUESKY_HANDLE"], st.secrets["BLUESKY_PASSWORD"])
    return client

supabase = init_supabase()
bsky_client = init_bsky_client()

def score_toxicity_via_api(text: str) -> float:
    """Hits the Hugging Face Inference API instead of local memory."""
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    payload = {"inputs": text[:512]}
    url = "https://api-inference.huggingface.co/models/unitary/toxic-bert"
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                for label_data in result[0]:
                    if label_data['label'] == 'toxic':
                        return round(label_data['score'], 4)
    except Exception:
        pass
    return 0.0

# --- 2. SIDEBAR & UI ---
st.sidebar.title("ToxGraph 🕸️")
st.sidebar.markdown("### Focal Structure Analysis")
st.sidebar.markdown("This tool queries the Bluesky AT Protocol, scores toxicity via serverless AI inference, and maps the amplification network.")

st.title("Discourse Velocity & Toxicity Tracker")
search_query = st.text_input("Enter Target Keyword or Hashtag (e.g., 'india', '#politics')", "")

# --- 3. DATA INGESTION ---
if st.button("Analyze Discourse", type="primary"):
    if not search_query:
        st.warning("Please enter a keyword.")
    else:
        with st.spinner(f"Querying Bluesky and running remote AI inference..."):
            try:
                response = bsky_client.app.bsky.feed.search_posts(
                    params={'q': search_query, 'limit': 30, 'lang': 'en'}
                )
                posts_data = response.posts
            except Exception as e:
                st.error(f"Bluesky SDK Error: {e}")
                st.stop()
            
            if not posts_data:
                st.error("No posts found for this keyword.")
                st.session_state.has_data = False
            else:
                G = nx.DiGraph()
                st.session_state.node_data.clear() 
                
                for item in posts_data:
                    author_data = item.author
                    record = item.record
                    text = getattr(record, 'text', '')
                    
                    if not text:
                        continue
                        
                    # Remote AI Call
                    tox_score = score_toxicity_via_api(text)
                    
                    author_handle = getattr(author_data, 'handle', 'Unknown')
                    author_did = getattr(author_data, 'did', '')
                    post_uri = getattr(item, 'uri', '')
                    
                    st.session_state.node_data[author_did] = {
                        "handle": author_handle,
                        "text": text,
                        "toxicity": tox_score
                    }
                    
                    reply_attr = getattr(record, 'reply', None)
                    parent_uri = getattr(reply_attr.parent, 'uri', None) if reply_attr and hasattr(reply_attr, 'parent') else None
                    
                    node_color = "#ff4b4b" if tox_score > 0.7 else "#4b8bff"
                    G.add_node(author_did, label=author_handle, color=node_color, size=20)
                    
                    if parent_uri:
                        parts = parent_uri.split('/')
                        if len(parts) >= 3 and parts[2].startswith('did:'):
                            target_did = parts[2]
                            G.add_edge(author_did, target_did)
                            if target_did not in G:
                                G.add_node(target_did, label="Target Account", color="#cccccc", size=15)
                    
                    try:
                        supabase.table('authors').upsert({"did": author_did, "handle": author_handle}).execute()
                        supabase.table('posts').upsert({
                            "post_uri": post_uri,
                            "author_did": author_did,
                            "created_at": getattr(record, 'created_at', None),
                            "content": text,
                            "toxicity_score": tox_score,
                            "reply_parent_uri": parent_uri,
                            "language": 'en'
                        }).execute()
                    except Exception:
                        pass
                
                centrality = nx.degree_centrality(G)
                st.session_state.graph_nodes = []
                st.session_state.graph_edges = []
                
                for n in G.nodes():
                    node_attr = G.nodes[n]
                    math_size = node_attr.get('size', 15) + (centrality.get(n, 0) * 150)
                    st.session_state.graph_nodes.append(Node(id=n, label=node_attr.get('label', n), size=math_size, color=node_attr.get('color', '#cccccc')))
                    
                for u, v in G.edges():
                    st.session_state.graph_edges.append(Edge(source=u, target=v, color="#666666", type="CURVE_SMOOTH"))
                
                st.session_state.has_data = True

# --- 4. GRAPH RENDERING ---
if st.session_state.has_data:
    st.subheader("Amplification Network")
    config = Config(width="100%", height=600, directed=True, physics=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
    
    clicked_node_id = agraph(nodes=st.session_state.graph_nodes, edges=st.session_state.graph_edges, config=config)

    # --- 5. SIDEBAR REACTIVITY ---
    if clicked_node_id and clicked_node_id in st.session_state.node_data:
        data = st.session_state.node_data[clicked_node_id]
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Node Intelligence")
        st.sidebar.markdown(f"**Actor:** `@{data['handle']}`")
        
        if data['toxicity'] > 0.7:
            st.sidebar.error(f"**Toxicity Score:** {data['toxicity']} (High Risk)")
        else:
            st.sidebar.success(f"**Toxicity Score:** {data['toxicity']} (Low Risk)")
            
        st.sidebar.markdown("**Content Payload:**")
        st.sidebar.info(data['text'])