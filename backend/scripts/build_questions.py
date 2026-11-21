"""
Builds eval_questions.jsonl: 147 retrieval questions over 6 repos.
Each question has the file(s) and line range(s) where the answer lives,
read from the repos at the commit that gets ingested.
Split: ~1/3 "dev" (allowed for tuning), ~2/3 "test" (only for the final number).
"""
import json
import random

Q = []


def q(repo, question, *spans, qtype="how"):
    Q.append({
        "repo": repo,
        "question": question,
        "type": qtype,
        "expected": [{"path": p, "lines": [a, b]} for p, a, b in spans],
    })


# ---------------------------------------------------------------- SudarshanLB (C++)
R = "puneetkr-06/SudarshanLB"
q(R, "How does the balancer decide which backend server a given user is sent to?",
  ("src/ConsistentHashRing.cpp", 41, 58), ("src/LoadBalancer.cpp", 150, 159))
q(R, "What happens when one of the backend servers stops responding?", ("src/LoadBalancer.cpp", 68, 89))
q(R, "How often are the backend servers checked for liveness?", ("src/LoadBalancer.cpp", 69, 72), qtype="config")
q(R, "How is it tested whether a server is alive?", ("src/LoadBalancer.cpp", 48, 66))
q(R, "How are TCP connections to backends reused across requests?", ("src/ConnectionPool.cpp", 41, 63))
q(R, "What does the client receive if no connection to the chosen backend can be made?",
  ("src/LoadBalancer.cpp", 164, 180))
q(R, "Which part of the HTTP request is used to keep a user on the same server?",
  ("src/LoadBalancer.cpp", 150, 157))
q(R, "How many virtual points does each server get on the ring by default?",
  ("include/ConsistentHashRing.h", 53, 55), ("src/ConsistentHashRing.cpp", 12, 21), qtype="config")
q(R, "How can an operator add or remove servers while the proxy is running?", ("src/main.cpp", 20, 35))
q(R, "How does it handle many client connections without one thread per connection?",
  ("src/LoadBalancer.cpp", 121, 142))
q(R, "How is the listening socket created and configured?", ("src/LoadBalancer.cpp", 96, 119))
q(R, "How is the ring kept safe when one thread reads it while another modifies it?",
  ("src/ConsistentHashRing.cpp", 12, 15), ("src/ConsistentHashRing.cpp", 41, 44), ("include/ConsistentHashRing.h", 45, 49))
q(R, "Why does the node struct define its own copy and move operations?", ("include/ConsistentHashRing.h", 8, 43), qtype="why")
q(R, "What happens when a user's hash is bigger than every point on the ring?", ("src/ConsistentHashRing.cpp", 46, 57))
q(R, "How do I start dummy backend servers for testing?", ("test/dummy_backends.py", 1, 16), qtype="setup")
q(R, "What throughput and tail latency were measured in benchmarks?", ("readme.md", 14, 22), qtype="docs")
q(R, "Why is the broken pipe signal ignored at startup?", ("src/main.cpp", 7, 9), ("readme.md", 19, 19), qtype="why")
q(R, "How is a recovered server put back into rotation?", ("src/LoadBalancer.cpp", 74, 81))
q(R, "Which hash function places keys on the ring?", ("src/ConsistentHashRing.cpp", 8, 10))
q(R, "How are pooled sockets cleaned up when the program shuts down?", ("src/ConnectionPool.cpp", 10, 17))

# ---------------------------------------------------------------- CUDA GEMM
R = "puneetkr-06/Custom_Cuda_Optimised_Matrix_Multiplication"
q(R, "Why is the simplest kernel slow?", ("gemm.cu", 70, 104), ("README.md", 66, 76), qtype="why")
q(R, "How is on-chip shared memory used to speed up the multiplication?", ("gemm.cu", 106, 193), ("README.md", 78, 96))
q(R, "Why are there two barriers inside the tile loop?", ("gemm.cu", 173, 186), ("README.md", 35, 36), qtype="why")
q(R, "How does the library-based version deal with row-major versus column-major storage?", ("gemm.cu", 195, 230))
q(R, "How are the GPU results checked for correctness?", ("gemm.cu", 232, 260))
q(R, "What error tolerance is used when comparing outputs?", ("gemm.cu", 235, 245), qtype="config")
q(R, "How is kernel execution time measured on the GPU?", ("gemm.cu", 304, 335))
q(R, "How many warm-up and timed runs are done per method?", ("gemm.cu", 309, 311), qtype="config")
q(R, "How is throughput in GFLOPS calculated?", ("gemm.cu", 280, 282), ("gemm.cu", 332, 336))
q(R, "Which matrix sizes are benchmarked?", ("gemm.cu", 454, 456), ("README.md", 46, 58), qtype="config")
q(R, "How is the GPU's theoretical memory bandwidth computed?", ("gemm.cu", 436, 448))
q(R, "What happens when the matrix size is not a multiple of the tile size?", ("gemm.cu", 159, 171), ("gemm.cu", 189, 192))
q(R, "How do I compile and run it on Google Colab?", ("README.md", 100, 114), qtype="setup")
q(R, "How can I check for shared memory bank conflicts with the profiler?", ("README.md", 139, 144), qtype="setup")
q(R, "What speedup did the tiled version achieve over the naive one?", ("README.md", 42, 60), qtype="docs")
q(R, "How are CUDA and cuBLAS API errors handled?", ("gemm.cu", 43, 62))
q(R, "Why does the tiled kernel avoid bank conflicts?", ("gemm.cu", 129, 137), ("README.md", 33, 34), qtype="why")

# ---------------------------------------------------------------- ViewerLens (Python ML + React)
R = "puneetkr-06/ViewerLens"
q(R, "How does the app fetch the comments of a YouTube video?", ("backend/app.py", 27, 60))
q(R, "What is the maximum number of comments analyzed per video?", ("backend/app.py", 27, 32), ("backend/app.py", 98, 100), qtype="config")
q(R, "How is the video ID extracted from a link?", ("backend/app.py", 22, 25))
q(R, "Which model labels comment sentiment in the API right now?", ("backend/app.py", 112, 114), ("backend/bert_model.py", 17, 30))
q(R, "How is the like percentage of a video computed?", ("backend/app.py", 63, 83))
q(R, "What does the analyze endpoint return to the frontend?", ("backend/app.py", 117, 135))
q(R, "How are near-duplicate spam comments removed from the training data?", ("backend/model.py", 20, 49), ("backend/train_svm.py", 18, 37))
q(R, "How was class imbalance handled during training?", ("backend/model.py", 76, 87), ("backend/train_svm.py", 61, 72))
q(R, "What text cleaning is done before training the classical models?", ("backend/model.py", 13, 18), ("backend/train_svm.py", 11, 16))
q(R, "Which pretrained transformer and hyperparameters are used for fine-tuning?", ("backend/train_bert.py", 17, 24), ("backend/train_bert.py", 106, 117), qtype="config")
q(R, "Where is the fine-tuned model saved?", ("backend/train_bert.py", 149, 155))
q(R, "How is the transformer model evaluated quickly on a sample?", ("backend/eval_bert.py", 12, 26))
q(R, "What accuracy and F1 did each approach reach?", ("backend/README.md", 1, 20), qtype="docs")
q(R, "Which classical model replaced Naive Bayes and with what settings?", ("backend/model.py", 106, 115))
q(R, "How are the TF-IDF features configured?", ("backend/model.py", 97, 101), ("backend/train_svm.py", 81, 86), qtype="config")
q(R, "How does the frontend call the backend and where is its URL set?", ("frontend/src/App.jsx", 10, 35))
q(R, "How can users filter comments by sentiment in the UI?", ("frontend/src/App.jsx", 37, 40), ("frontend/src/App.jsx", 170, 195))
q(R, "What happens when the YouTube API returns an error?", ("backend/app.py", 43, 47), ("backend/app.py", 109, 110))
q(R, "Is there a way to download comments without the official API?", ("backend/comments.py", 1, 29))
q(R, "Which port does the backend server run on?", ("backend/app.py", 137, 138), qtype="config")
q(R, "How are numeric class IDs mapped to sentiment names?", ("backend/bert_model.py", 11, 15), ("backend/train_bert.py", 26, 32))
q(R, "How is the like ratio drawn as a circle on the page?", ("frontend/src/App.jsx", 117, 160))

# ---------------------------------------------------------------- Muysnc (Node + React)
R = "puneetkr-06/Muysnc"
q(R, "How does mood-based song suggestion work end to end?", ("Backend/controllers/gemini_controller.js", 83, 225))
q(R, "What happens if the AI call fails while suggesting songs for a mood?", ("Backend/controllers/gemini_controller.js", 147, 160), ("Backend/controllers/gemini_controller.js", 4, 12))
q(R, "How is a free-text mood mapped to a fixed category for fallback songs?", ("Backend/controllers/gemini_controller.js", 14, 24))
q(R, "Which generation settings are used for the AI request?", ("Backend/controllers/gemini_controller.js", 37, 54), qtype="config")
q(R, "How does the backend cope when the AI response is not clean JSON?", ("Backend/controllers/gemini_controller.js", 111, 145))
q(R, "Where do the playable audio links in search results come from?", ("Backend/controllers/search_controller.js", 18, 33), ("Backend/controllers/aisearch_controller.js", 15, 63))
q(R, "How many search results are returned at most?", ("Backend/controllers/search_controller.js", 20, 20), ("Backend/controllers/aisearch_controller.js", 24, 24), qtype="config")
q(R, "How long are trending and chart results cached?", ("Backend/controllers/trending_controller.js", 4, 10), ("Backend/controllers/topCharts_controller.js", 4, 10), ("Backend/controllers/biggestHitsController.js", 4, 10), qtype="config")
q(R, "What happens if Spotify data cannot be fetched for the top charts?", ("Backend/controllers/topCharts_controller.js", 41, 54))
q(R, "Which artists appear in the artist spotlight and how are their images fetched?", ("Backend/controllers/artistController.js", 12, 51))
q(R, "How is a song added to a user's listening history and how many are kept?", ("Backend/controllers/recentlyPlayed_controller.js", 5, 30))
q(R, "How does login handle a first-time user who has no account yet?", ("Backend/controllers/user_controller.js", 24, 48))
q(R, "What information is stored for each user in the database?", ("Backend/models/user.js", 3, 35))
q(R, "Which frontend origins are allowed to call the API?", ("Backend/app.js", 6, 11), qtype="config")
q(R, "How does the server connect to the database?", ("Backend/db/db.js", 1, 13))
q(R, "Which URL paths does the music API expose?", ("Backend/routes/spotify_auth_route.js", 14, 20))
q(R, "How are JWT tokens verified for protected routes?", ("Backend/middlewares/user_auth.js", 4, 35))
q(R, "How does the frontend make sure only one song plays at a time?", ("Frontend/src/utils/audioPlayer.js", 1, 17))
q(R, "How does the frontend keep the user signed in after a page refresh?", ("Frontend/src/firebase/firebase.js", 13, 18), ("Frontend/src/firebase/AuthContext.jsx", 11, 31))
q(R, "Where is the backend base URL configured on the frontend?", ("Frontend/src/utils/config.js", 1, 2), qtype="config")
q(R, "How does the AI suggestion page block users who are not logged in?", ("Frontend/src/pages/Moodsync/Moodsync.jsx", 10, 26))
q(R, "How does Google sign-in work on the login page?", ("Frontend/src/pages/Login/Login.jsx", 56, 78))
q(R, "When does the player record a song as recently played?", ("Frontend/src/components/Playbar/Playbar.jsx", 43, 63), ("Frontend/src/utils/updateRecentlyPlayed.js", 5, 27))
q(R, "What pages exist in the frontend app?", ("Frontend/src/App.jsx", 18, 34))
q(R, "What does the health check endpoint report?", ("Backend/app.js", 25, 33))

# ---------------------------------------------------------------- Rentora (TypeScript)
R = "puneetkr-06/Rentora"
P = "backend/src/controllers/"
q(R, "How are monthly rent invoices generated for a lease?", (P + "paymentController.ts", 192, 242))
q(R, "When a tenant pays rent, which invoice gets marked as paid?", (P + "paymentController.ts", 6, 49))
q(R, "Which cached data is cleared after a rent payment?", (P + "paymentController.ts", 51, 81))
q(R, "How long is dashboard data kept in the cache?", (P + "propertyController.ts", 63, 64), (P + "paymentController.ts", 147, 147), (P + "leaseController.ts", 131, 132), qtype="config")
q(R, "How does a tenant join a property using a code?", (P + "leaseController.ts", 62, 105))
q(R, "How is the code that tenants use to join generated?", (P + "leaseController.ts", 6, 8), (P + "leaseController.ts", 28, 28))
q(R, "What stops two tenants from claiming the same join code?", (P + "leaseController.ts", 71, 78))
q(R, "How can an owner assign an already registered tenant directly to a room?", (P + "leaseController.ts", 140, 181))
q(R, "How is the amount an owner collected this month calculated?", (P + "paymentController.ts", 315, 328))
q(R, "How is the occupancy percentage of each property computed?", (P + "propertyController.ts", 95, 145))
q(R, "How is revenue per property computed in the property list?", (P + "propertyController.ts", 52, 61))
q(R, "What happens to tenants when a room is deleted?", (P + "roomController.ts", 138, 169))
q(R, "How does an owner vacate a room?", (P + "roomController.ts", 99, 136))
q(R, "How does the server verify that a new room belongs to the owner's property?", (P + "roomController.ts", 14, 25))
q(R, "How are API requests authenticated on the server?", ("backend/src/middleware/authMiddleware.ts", 9, 35))
q(R, "Why does login verify the password with a separate temporary client?", (P + "authController.ts", 55, 69), qtype="why")
q(R, "How are new accounts created and is email confirmation needed?", (P + "authController.ts", 5, 46))
q(R, "Is there a length limit on notices posted by owners?", ("backend/src/controllers/noticeController.ts", 15, 17), qtype="config")
q(R, "After an owner posts a notice, how do tenants see it despite caching?", (P + "noticeController.ts", 39, 52))
q(R, "How does a tenant get the notices for the properties they live in?", (P + "noticeController.ts", 60, 110))
q(R, "How does a tenant raise a maintenance complaint?", (P + "complaintController.ts", 19, 56))
q(R, "How are complaint records reshaped before being sent to the frontend?", (P + "complaintController.ts", 5, 17))
q(R, "Which profile fields can a user update and which one is mandatory?", (P + "userController.ts", 21, 80))
q(R, "What happens in the browser when the login session expires?", ("frontend/components/FetchInterceptor.tsx", 5, 34))
q(R, "How is the cache client configured and how are keys invalidated?", ("backend/src/utils/redis.ts", 1, 19))
q(R, "What does the health check endpoint verify?", ("backend/src/index.ts", 41, 54))
q(R, "Which API route groups does the backend expose?", ("backend/src/index.ts", 31, 39))
q(R, "How are a tenant's pending dues computed?", (P + "paymentController.ts", 244, 272))
q(R, "How does an owner see complaints across all their properties?", (P + "complaintController.ts", 74, 99))
q(R, "What are the two kinds of users and what can each one do?", ("README.md", 6, 17), qtype="docs")
q(R, "How is a tenant's payment history fetched?", (P + "paymentController.ts", 155, 190))
q(R, "What details about tenants does an owner see for each room?", (P + "roomController.ts", 72, 93))

# ---------------------------------------------------------------- RepoTrace (this project)
R = "puneetkr-06/RepoTrace"
B = "backend/"
q(R, "How does the app download a repository's source code?", (B + "github_source.py", 122, 176))
q(R, "Why is the repository downloaded as one archive instead of file by file?", (B + "github_source.py", 1, 6), qtype="why")
q(R, "What stops someone from indexing a private repository?", (B + "github_source.py", 81, 85))
q(R, "Which folders and file types are skipped during ingestion?", (B + "github_source.py", 19, 28), (B + "github_source.py", 114, 120), qtype="config")
q(R, "How are binary or oversized files handled?", (B + "github_source.py", 152, 166))
q(R, "How is source code split into pieces for search?", (B + "chunking.py", 47, 62), (B + "chunking.py", 85, 110))
q(R, "How are line numbers worked out for each piece of code?", (B + "chunking.py", 80, 110))
q(R, "Why is the file path added to the text that gets embedded?", (B + "chunking.py", 73, 77), qtype="why")
q(R, "How does the server check that a repository fits the free embedding budget?", (B + "ingest.py", 72, 101))
q(R, "How does the app know that indexing finished successfully?", (B + "ingest.py", 1, 12), (B + "ingest.py", 127, 141), (B + "vector_store.py", 146, 160))
q(R, "How does the system stay under the embedding provider's per-minute token limit?", (B + "vector_store.py", 72, 96))
q(R, "What happens when the embedding API returns a rate limit error?", (B + "vector_store.py", 99, 125))
q(R, "Where is the monthly embedding token usage stored?", (B + "vector_store.py", 163, 189))
q(R, "How does the frontend show indexing progress?", ("frontend/src/pages/RepoChat.tsx", 61, 112), ("frontend/src/pages/RepoChat.tsx", 156, 192))
q(R, "How often does the frontend check indexing status, and why that often?", ("frontend/src/pages/RepoChat.tsx", 8, 12), qtype="config")
q(R, "How does the frontend read the streamed answer?", ("frontend/src/api.ts", 57, 89))
q(R, "How does the chat endpoint stream its answer?", (B + "main.py", 118, 169))
q(R, "How are citations with line ranges built?", (B + "rag.py", 106, 131))
q(R, "When is the previous question included in the search query?", (B + "rag.py", 60, 89))
q(R, "How is chat history limited before it is sent to the model?", (B + "rag.py", 48, 57), (B + "config.py", 106, 108))
q(R, "What instructions does the model get for answering?", (B + "rag.py", 15, 30))
q(R, "How are requests limited per user and globally?", (B + "rate_limit.py", 15, 41), (B + "main.py", 131, 134))
q(R, "How is the real client IP found behind the hosting proxy?", (B + "rate_limit.py", 44, 50))
q(R, "Why is the model's thinking turned off?", (B + "config.py", 97, 101), qtype="why")
q(R, "How was the per-repository token limit derived?", (B + "config.py", 74, 78), qtype="why")
q(R, "What does the metrics endpoint report?", (B + "metrics.py", 42, 76))
q(R, "How are percentiles computed for latency numbers?", (B + "metrics.py", 21, 28))
q(R, "What happens if a repository is requested while it is already being indexed?", (B + "main.py", 61, 87), (B + "ingest.py", 55, 63))
q(R, "Which websites are allowed to call the API?", (B + "config.py", 32, 37), (B + "main.py", 22, 28), qtype="config")
q(R, "How do I run the project on my own machine?", ("README.md", 88, 105), qtype="setup")
q(R, "Which programming languages get syntax-aware splitting?", (B + "chunking.py", 20, 43), qtype="config")

# ---------------------------------------------------------------- split + write
rng = random.Random(7)  # fixed seed: the split never changes between runs
for i, row in enumerate(Q):
    row["id"] = f"q{i + 1:03d}"
    row["split"] = "dev" if rng.random() < 1 / 3 else "test"

with open("eval_questions.jsonl", "w") as f:
    for row in Q:
        f.write(json.dumps(row) + "\n")

from collections import Counter
print(len(Q), "questions")
print(Counter(r["repo"].split("/")[1] for r in Q))
print(Counter(r["split"] for r in Q))
print(Counter(r["type"] for r in Q))
