import streamlit as st
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
from supabase import create_client
from atproto import Client as BskyClient
import requests
import time

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

# --- 1. CLIENT INITIALIZATION ---
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

# --- 2. FAIL-FAST MULTI-LABEL TOXICITY INFERENCE ---
def score_toxicity_batch(texts: list[str]) -> list[dict]:
    """
    Sends batch inference requests to Hugging Face.
    Evaluates multilabel risks (toxic, identity_hate, insult, severe_toxic, threat).
    """
    if not texts:
        return []

    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    url = "https://api-inference.huggingface.co/models/unitary/toxic-bert"
    payload = {"inputs": [t[:512] for t in texts]}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            
            # Handle model cold-start
            if response.status_code == 503:
                wait_time = response.json().get("estimated_time", 10.0)
                st.toast(f"Model cold-starting. Retrying in {int(wait_time)}s...", icon="⏳")
                time.sleep(min(wait_time, 15.0))
                continue
                
            if response.status_code == 429:
                st.error("Hugging Face API Rate Limit Exceeded.")
                st.stop()
                
            if response.status_code != 200:
                st.error(f"HF Inference Error ({response.status_code}): {response.text}")
                st.stop()
                
            result = response.json()
            
            # Normalize batch response to extract aggregate toxicity and top category
            scored_batch = []
            for item in result:
                if isinstance(item, list):
                    scores = {label_obj['label']: round(label_obj['score'], 4) for label_obj in item}
                    # Aggregate highest risk indicator (catches slurs under identity_hate/insult)
                    top_score = max(scores.values()) if scores else 0.0
                    top_label = max(scores, key=scores.get) if scores else "unknown"
                    scored_batch.append({
                        "max_score": top_score,
                        "primary_flag": top_label,
                        "breakdown": scores
                    })
                else:
                    scored_batch.append({"max_score": 0.0, "primary_flag": "error", "breakdown": {}})
            return scored_batch

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                st.error(f"Inference Connection Failure: {str(e)}")
                st.stop()
            time.sleep(2)
            
    return [{"max_score": 0.0, "primary_flag": "timeout", "breakdown": {}} for _ in texts]

# --- 3. SIDEBAR & UI ---
st.sidebar.title("ToxGraph 🕸️")
st.sidebar.markdown("### Focal Structure Analysis")
st.sidebar.markdown("Queries the Bluesky AT Protocol, computes multi-label toxicity vectors via remote BERT inference, and models network amplification dynamics.")

st.title("Discourse Velocity & Toxicity Tracker")
search_query = st.text_input("Enter Target Keyword or Hashtag (e.g., 'india', '#politics')", "")

# --- 4. PIPELINE EXECUTION ---
if st.button("Analyze Discourse", type="primary"):
    if not search_query.strip():
        st.warning("Please enter a valid keyword.")
    else:
        with st.spinner("Fetching network data from Bluesky and scoring toxicity..."):
            try:
                response = bsky_client.app.bsky.feed.search_posts(
                    params={'q': search_query, 'limit': 30, 'lang': 'en'}
                )
                posts_data = response.posts
            except Exception as e:
                st.error(f"Bluesky AT Protocol Error: {str(e)}")
                st.stop()
            
            if not posts_data:
                st.error("No posts found for this keyword.")
                st.session_state.has_data = False
            else:
                G = nx.DiGraph()
                st.session_state.node_data.clear()
                
                # Extract text for batch inference
                valid_posts = []
                texts_to_score = []
                for item in posts_data:
                    text = getattr(item.record, 'text', '')
                    if text.strip():
                        valid_posts.append(item)
                        texts_to_score.append(text)
                
                # Execute batch scoring
                inference_results = score_toxicity_batch(texts_to_score)
                
                # Build graph nodes and relationships
                for item, score_data in zip(valid_posts, inference_results):
                    author_data = item.author
                    record = item.record
                    text = getattr(record, 'text', '')
                    author_handle = getattr(author_data, 'handle', 'Unknown')
                    author_did = getattr(author_data, 'did', '')
                    post_uri = getattr(item, 'uri', '')
                    
                    tox_score = score_data["max_score"]
                    primary_flag = score_data["primary_flag"]
                    
                    st.session_state.node_data[author_did] = {
                        "handle": author_handle,
                        "text": text,
                        "toxicity": tox_score,
                        "primary_flag": primary_flag,
                        "breakdown": score_data["breakdown"],
                        "type": "source"
                    }
                    
                    # Severe/High Risk = Red, Medium = Orange, Clean = Blue
                    if tox_score > 0.7:
                        node_color = "#ff4b4b"
                    elif tox_score > 0.4:
                        node_color = "#ffa500"
                    else:
                        node_color = "#4b8bff"
                        
                    G.add_node(author_did, label=author_handle, color=node_color, size=20)
                    
                    # Parse parent reply references
                    reply_attr = getattr(record, 'reply', None)
                    parent_uri = getattr(reply_attr.parent, 'uri', None) if reply_attr and hasattr(reply_attr, 'parent') else None
                    
                    if parent_uri:
                        parts = parent_uri.split('/')
                        if len(parts) >= 3 and parts[2].startswith('did:'):
                            target_did = parts[2]
                            G.add_edge(author_did, target_did)
                            
                            if target_did not in G:
                                G.add_node(target_did, label=f"Target ({target_did[:12]}...)", color="#888888", size=15)
                                st.session_state.node_data[target_did] = {
                                    "handle": target_did,
                                    "text": "[Target account reference - original parent post not in current search window]",
                                    "toxicity": 0.0,
                                    "primary_flag": "unscanned",
                                    "breakdown": {},
                                    "type": "target"
                                }
                    
                    # Persist metadata to Supabase
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
                    except Exception as db_err:
                        st.sidebar.warning(f"Database sync warning: {str(db_err)}")
                
                # Topological math for node sizing
                if len(G.nodes()) > 0:
                    centrality = nx.degree_centrality(G)
                    st.session_state.graph_nodes = []
                    st.session_state.graph_edges = []
                    
                    for n in G.nodes():
                        node_attr = G.nodes[n]
                        math_size = node_attr.get('size', 15) + (centrality.get(n, 0) * 120)
                        st.session_state.graph_nodes.append(
                            Node(id=n, label=node_attr.get('label', n), size=math_size, color=node_attr.get('color', '#cccccc'))
                        )
                        
                    for u, v in G.edges():
                        st.session_state.graph_edges.append(
                            Edge(source=u, target=v, color="#555555", type="CURVE_SMOOTH")
                        )
                    
                    st.session_state.has_data = True

# --- 5. VISUALIZATION & INTERACTION ---
if st.session_state.has_data and st.session_state.graph_nodes:
    st.subheader("Amplification Network")
    config = Config(
        width="100%", 
        height=600, 
        directed=True, 
        physics=True, 
        nodeHighlightBehavior=True, 
        highlightColor="#F7A7A6"
    )
    
    clicked_node_id = agraph(nodes=st.session_state.graph_nodes, edges=st.session_state.graph_edges, config=config)

    if clicked_node_id and clicked_node_id in st.session_state.node_data:
        data = st.session_state.node_data[clicked_node_id]
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Node Intelligence")
        st.sidebar.markdown(f"**Actor:** `@{data['handle']}`")
        
        if data["type"] == "target":
            st.sidebar.info("Target node referenced in reply. Not indexed in current query batch.")
        else:
            if data['toxicity'] > 0.7:
                st.sidebar.error(f"**Risk Level:** Critical ({data['toxicity']}) — `{data['primary_flag']}`")
            elif data['toxicity'] > 0.4:
                st.sidebar.warning(f"**Risk Level:** Moderate ({data['toxicity']}) — `{data['primary_flag']}`")
            else:
                st.sidebar.success(f"**Risk Level:** Low ({data['toxicity']})")
                
            st.sidebar.markdown("**Content Payload:**")
            st.sidebar.info(data['text'])
            
            with st.sidebar.expander("Vector Label Breakdown"):
                st.json(data['breakdown'])