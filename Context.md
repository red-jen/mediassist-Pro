Contexte du projet
Vous êtes ingénieur en intelligence artificielle au sein d’une entreprise spécialisée dans la distribution d’équipements biomédicaux. Les laboratoires clients rencontrent régulièrement des difficultés techniques avec leurs équipements. L’accès aux manuels techniques, souvent volumineux et dispersés en plusieurs PDF, est chronophage, et l’intervention du service technique engendre des délais coûteux, impactant la continuité des analyses et la remise des résultats aux cliniciens.

Votre mission consiste à concevoir et déployer une architecture RAG (Retrieval-Augmented Generation) optimisée, capable d’indexer les manuels techniques et les bases de connaissances internes afin de fournir aux techniciens de laboratoire des réponses précises, actionnables et sourcées, en langage naturel, avant même l’ouverture d’un ticket support.

RAG
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
Framework : FastAPI (API REST asynchrone)
Validation : Pydantic
ORM : SQLAlchemy
Construction d’un pipeline RAG: LangChain
Authentification : JWT
Base de données : PostgreSQL
Configuration : pydantic-settings + fichiers .env
Conteneurisation : Docker + Docker Compose
Gestion centralisée des exceptions
Tests unitaires
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
Pipeline CI/CD → Kubernetes :
Exécuter les tests (code + RAG)
Construire l’image Docker
Publier l’image sur Docker Hub
Déployer l’image dans Kubernetes (un seul pod)
Superviser et gérer le Pod via Minikube
Monitoring avec Prometheus & Grafana:
Surveiller les métriques du RAG et du Pod
Collecter les métriques d’infrastructure (CPU, RAM, statut du Pod)
Collecter les métriques applicatives du RAG (latence, qualité des réponses, erreurs, nombre de requêtes)
Configurer des alertes basées sur les métriques du RAG: Définir des seuils pour la latence, les erreurs ou la qualité des réponses
Modalités pédagogiques
Travail : individuel

Durée : 10 jours

Période : Du 02/02/2026 au 13/02/2026 avant minuit.

Modalités d'évaluation
Mise en situation
Code review
Culture du projet
Livrables
Dépôt GitHub contenant :
Code source complet
README.md (description, installation, technologies)
Critères de performance
- Précision du RAG : qualité et pertinence des réponses générées, fondées exclusivement sur les documents indexés, avec réduction des hallucinations.

- Qualité du prompt engineering : clarté, cohérence et efficacité des prompts pour guider le LLM vers des réponses fiables, contextualisées et actionnables.

- Sécurité et fiabilité : authentification JWT, gestion robuste des erreurs, protection contre les fuites de données et respect des bonnes pratiques de sécurité.

- Qualité du code : couverture de tests ≥ 80 %, architecture modulaire et maintenable, documentation complète, et conteneurisation opérationnelle (Docker).
Situation professionnelle
Optimisation continue et déploiement fiable de modèles IA avec des pipelines MLOps
Besoin visé ou problème rencontré
Une entreprise a mis en production un modèle d’IA(prédiction,recommandation,classification),mais rencontre des problèmes de dérive,d’actualisation des données ou de fiabilité du système.Elle souhaite automatiser le suivi,le ré-entraînement et le déploiement des modèles pour garantir leur performance dans le temps. Il est donc demandé à l’équipe IA de mettre en œuvre une approche MLOps permettant la maintenance,la supervision et l’évolution continue des modèles dans un environnement de production
30 compétences visées