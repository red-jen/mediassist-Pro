Contexte du projet
ProtoCare est une startup innovante du secteur de la santé qui développe une solution d’intelligence artificielle destinée à standardiser la prise en charge des patients et à assister les professionnels de santé dans leurs décisions cliniques. La solution repose sur un système de RAG permettant d’automatiser l’accès aux protocoles médicaux et de fournir aux médecins des réponses précises, fiables et contextualisées, facilitant un diagnostic rapide et une prise de décision éclairée, notamment en situation d’urgence.

Votre mission :

En tant que Développeur IA, vous serez en charge de concevoir, développer et déployer cette solution RAG, en veillant à son efficacité, sa fiabilité et son adéquation aux besoins des professionnels de santé.

Retrieval-Augmented Generation (RAG)
Prétraitement et Chunking
Importer le manuel technique ou les documents PDF de référence.
Choisir une méthode de chunking qui préserve le maximum de contexte dans chaque chunk.
Chaque chunk doit être accompagné de métadonnées utiles
Indexation et persistance des embeddings
Choisir une base de données vectorielle adaptée (ChromaDB, FAISS, Qdrant).
Sélectionner un modèle d’embeddings (Hugging Face, Ollama)
Persister les embeddings dans un store vectoriel.
Retrieval (Récupération des informations)
Configurer un retriever qui permet de rechercher les chunks pertinents selon la requête de l’utilisateur.
Intégrer des techniques d’amélioration de la recherche (Query expansion, Reranking).
Génération de réponse (RAG)
Définir un prompt centralisé et bien formulé
Utiliser un LLM pour générer des réponses à partir des chunks récupérés
S’assurer que les réponses soient précises
Tables principales:
users: id, username, email, hashed_password, role
Query: id, query, reponse, created_at
Back-end:
Framework : FastAPI (Login, query, history, health)
Validation : Pydantic
ORM : SQLAlchemy
Construction d’un pipeline RAG: LangChain
Authentification : JWT (login)
Base de données : PostgreSQL
Configuration : pydantic-settings + fichiers .env
Conteneurisation : Docker + Docker Compose
Gestion centralisée des exceptions
Tests unitaires
Front-end (Streamlit / React):
Concevoir et développer une interface intuitive et professionnelle permettant aux médecins d’interagir rapidement avec l’assistant médical RAG.
Développer un Dashboard permettant d’afficher l’historique des requêtes et des interactions de chaque médecin.
LLMOps
MlFlow:

Logger la configuration RAG:
Chunking : Taille, overlap et stratégie de segmentation des documents sources
Embedding : Choix du modèle d'embedding, dimensionnalité et normalisation
Retrieval : Algorithme de similarité (cosine, L2), nombre de chunks retournés (k), et re-ranking
Logger les hyperparamètres de configuration du LLM (template de prompt, température, modèle sélectionné, maxtokens, topp et top_k).
Logger les réponses & contextes
Logger les métriques RAG ( Answer relevance, Faithfulness, Precision@k, Recall@k) avec DeepEval
Logger le pipeline RAG (modèle) avec LangChain
Pipeline CI/CD :
Exécuter les tests (code + RAG)
Construire l’image Docker
Publier l’image sur Docker Hub
Monitoring avec Prometheus & Grafana:
Surveiller les métriques du RAG et du container
Collecter les métriques d’infrastructure (CPU, RAM)
Collecter les métriques applicatives du RAG (latence, qualité des réponses, erreurs, nombre de requêtes)
Configurer des alertes basées sur les métriques du RAG: Définir des seuils pour la latence, les erreurs ou la qualité des réponses
Modalités pédagogiques
Brief : Jury Blanc

Travail individuel

Temporalité : 5 jours

Période : Du 23/02/2026 au 27/02/2026 à 23h59

Modalités d'évaluation
Date de soutenance (02/03/2026)
Durée de soutenance jury blanc (45 min), organisée comme suit : :
Présentation rapide (10 minutes) : Démonstration du contenu et des fonctionnalités principales de l’application.
Analyse du code (10 minutes) : Montrez et expliquez le code source, en soulignant vos choix techniques.
Mise en situation (20 minutes) : L’examinateur propose des cas d’usage pour tester votre application.
Code Review et questions techniques (5 minutes).
Livrables
Présentation pptx
Code Source(Github)
Dashboard de monitoring
Planification et gestion du projet
Un bon README + Architecture de la solution
Critères de performance
Qualité du code: Lisibilité, Documentation, Gestion des exceptions et Tests unitaires
Backend & Base de données: API FastAPI, Validation Pydantic, ORM SQLAlchemy, Auth JWT et PostgreSQL fonctionnelle
Pipeline RAG: Prétraitement & Chunking, Embeddings & Stockage vectoriel, Retrieval optimisé et Génération LLM cohérente
Pipeline LLMOps: Tracking des paramètres, Logging des réponses, Versioning du pipeline et CI/CD
Monitoring infrastructure : dashboards Prometheus/Grafana opérationnels, métriques pertinentes collectées.