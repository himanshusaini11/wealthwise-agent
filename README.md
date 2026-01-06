# WealthWise: Autonomous AI Financial Agent

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)  ![Docker](https://img.shields.io/badge/docker-production-blue) ![AWS](https://img.shields.io/badge/AWS-EC2-orange) ![Python](https://img.shields.io/badge/Python-3.11-yellow)

**WealthWise** is a stateful AI agent capable of analyzing financial transaction data and forecasting future spending trends. Unlike standard chatbots, it uses **LangGraph** to maintain conversation state and executes custom Python tools for precise data analysis.

The system is deployed on **AWS EC2** using a fully automated **CI/CD pipeline** with self-healing capabilities.

---

## System Architecture

I designed this system to mimic a production-grade ML environment, moving away from "notebook code" to robust, containerized microservices.

```text
+-----------------------------------------------------------------------------------------------+
|                            WEALTHWISE AGENT: SYSTEM ARCHITECTURE                              |
+-----------------------------------------------------------------------------------------------+

        1. DEVELOPMENT                  2. CI/CD PIPELINE                3. PRODUCTION (AWS Cloud)
   (Local Environment)               (GitHub Actions)                  (EC2 Ubuntu t2.micro)

 +---------------------+          +---------------------+          +----------------------------+
 | 💻 VS Code (Mac)    |          | 🐙 GitHub Repo      |          | ☁️  AWS EC2 Instance       |
 |                     |          |                     |          |                            |
 |  • Code Logic       |   git    |  • Secrets (ENV)    |   SSH    |  +----------------------+  |
 |  • Pydantic Schemas |   push   |  • Workflow:        | Deploy   |  | 🐳 Docker Container  |  |
 |  • Unit Tests       |--------->|    1. Login to AWS  |--------->|  |                      |  |
 +---------------------+          |    2. Disk Check    |          |  |  [Streamlit UI]      |  |
                                  |    3. Auto-Deploy   |          |  |        |             |  |
                                  +---------------------+          |  |        v             |  |
                                                                   |  |  [LangGraph Brain]   |  |
                                                                   |  |   /          \       |  |
      4. USER INTERACTION                                          |  |  /            \      |  |
                                                                   |  |[Python Tool] [LLM]   |  |
 +---------------------+          +---------------------+          |  | (Pandas/ML) (Groq)   |  |
 | 👤 User (Browser)   |  HTTP    | 🛡️ Firewall         |          |  +----------------------+  |
 |                     | Request  | (Security Group)    |          +----------------------------+
 | "Forecast my rent"  |--------->| Port 8501           |                        ^
 +---------------------+          +----------+----------+                        |
                                             |                                   |
                                             +-----------------------------------+
```


## Key Features
1. Stateful AI Agent (LangGraph)
    - Uses a graph-based orchestration engine (LangGraph) instead of simple linear chains.
    - Maintains memory of past interactions (e.g., "Forecast my spending" -> "Why is it so high?" -> Context preserved).

2. Robust Tool Engineering (Pydantic V2)
    - Problem: LLMs often hallucinate input formats (e.g., sending "fortnight" instead of 14).
    - Solution: Implemented Pydantic Validators (@field_validator) to intercept LLM inputs. The system automatically translates natural language (e.g., "next month", "2 weeks") into integers before the tool executes, preventing recursion loops and API rejections.

3. Production CI/CD Pipeline
    - GitHub Actions: Automatically deploys to AWS EC2 on every push to main.
    - Disk Management: Custom shell scripts monitor disk usage on the t2.micro instance, automatically pruning old Docker images to prevent storage crashes.
    - Alerting: Sends email notifications if deployment fails or disk space is critical.

4. Infrastructure as Code
    - Docker Compose: Orchestrates the application dependencies.
    - Environment Locking: Uses requirements.txt with pinned versions to eliminate environment drift between Local and Production.

## Tech Stack
- LLM: Llama-3-70b (via Groq API) / Gemini Flash
- Frameworks: LangChain, LangGraph, Pandas, Scikit-Learn
- Validation: Pydantic V2
- DevOps: Docker, AWS EC2, GitHub Actions, Bash Scripting

## How to Run Locally
1. Clone the repository

```bash
git clone [https://github.com/himanshusaini11/wealthwise-agent.git](https://github.com/himanshusaini11/wealthwise-agent.git)
cd wealthwise-agent
```

2. Set up Environment Variables Create a .env file:

``` bash
GROQ_API_KEY=your_key_here
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET_NAME=your_bucket
```
3. Run with Docker

```bash
docker-compose up --build
```

4. Access the App Open http://localhost:8501 in your browser.

## Future Improvements
- Multi-Modal Inputs: Allow users to upload bank statement PDFs.
- RAG Integration: Connect to a vector database to search through financial literacy documents.