# Surveillance CCTV

Application web de surveillance par traitement d'image. Le navigateur envoie les images, le serveur estime un fond, détecte le mouvement, limite l'alerte à une zone dessinée et enregistre une photographie lorsqu'un événement persiste.

Projet de Master 1 Génie Informatique, Université Mapon, cours Digital Images Processing. Encadrement : CT Hervek Kabengele Kabukala.

Le rapport et les diapositives sont dans [`latex/`](latex/). Les diagrammes UML sont des sources PlantUML dans [`latex/diagrammes/`](latex/diagrammes/).

## Lancer en local

Python 3.11 ou plus récent.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Ouvrir [http://127.0.0.1:8000](http://127.0.0.1:8000). La première visite demande de créer un compte. Le mot de passe est stocké par PBKDF2-HMAC-SHA256. La session est un cookie signé, valable 30 jours.

Une fenêtre OpenCV, sans le site, sert à vérifier la même chaîne :

```bash
python main.py              # webcam
python main.py video.mp4    # fichier
```

| Touche | Effet |
| --- | --- |
| `q` | Quitter |
| `r` | Réapprendre le fond |
| `o` | Otsu ou seuil manuel |
| `+` / `-` | Régler le seuil |
| `m` | Filtre médian ou gaussien |
| `z` | Dessiner une zone |
| `c` | Effacer les zones |

Clic gauche : sommet du polygone. Clic droit : fermer la zone.

## Interface

- **Caméra**, **Vidéo** ou **Démo**. Sur téléphone, le bouton de bascule choisit la caméra arrière.
- **Flux distant** : une adresse `http`, `https` ou `rtsp` joignable depuis la machine qui exécute le serveur. Une caméra du réseau local n'est pas joignable si le serveur est hébergé ailleurs.
- **Détection** : réapprendre le fond, seuil d'Otsu ou seuil manuel de 5 à 80, filtre médian.
- **Zone** : dessiner, fermer, effacer. Seul un objet dont le centroïde est dans le polygone peut déclencher une alerte.
- **Alertes** : vignettes et lien vers le journal CSV. Une photographie n'est servie qu'au compte qui l'a produite.

Le cadre orange marque le mouvement. Le cadre rouge marque un visage frontal reconnu dans une boîte en mouvement, après l'apprentissage du fond.

## Traitement

Chaque image est ramenée à 640 pixels de large, puis passe par `pipeline.py` :

1. niveaux de gris, flou gaussien 5×5 ou médian 5 ;
2. égalisation d'histogramme seulement si la scène est durablement sombre ;
3. fond par moyenne glissante, `B_t = (1 - α) B_{t-1} + α I_t`, appris sur 45 images puis mis à jour avec α = 0,02 hors des pixels de premier plan ;
4. différence absolue, seuil d'Otsu ou seuil manuel, ouverture morphologique 3×3 ;
5. contours externes, aire minimale, boîte et centroïde.

Une alerte de mouvement attend 12 images consécutives. L'obscurcissement et le déplacement de caméra sont mesurés avant la morphologie : un voile sombre ou un décalage de toute l'image ne doit pas être pris pour un objet. La cascade de Haar (`haarcascade_frontalface_default`) ne regarde que les premières boîtes en mouvement dans la zone.

## Données

| Variable | Rôle | Défaut en local |
| --- | --- | --- |
| `DATA_DIR` | Comptes SQLite et secret de session | répertoire du projet |
| `ALERTS_DIR` | Photographies et journaux | `DATA_DIR/alerts` |
| `SESSION_SECRET` | Signature des cookies | fichier `DATA_DIR/.session_secret` |
| `PORT` | Port d'écoute | `8000` |

`users.db`, `.session_secret`, `.env` et le dossier `alerts/` sont ignorés par Git. Changer `SESSION_SECRET` déconnecte tout le monde.

## Déploiement

Le conteneur est décrit par le `Dockerfile`. L'image écoute le port 8000 et monte les données sur `/data`.

**Render.** Le fichier `render.yaml` crée un service Docker, un disque de 1 Go monté sur `/var/data`, et génère `SESSION_SECRET`. Le plan `starter` est payant : le plan gratuit s'endort et n'a pas de disque. Santé : `GET /api/health`.

**Railway.** Le fichier `railway.toml` construit le même Dockerfile et vérifie `/api/health`. Dans le tableau de bord, ajouter un volume monté sur `/data` et une variable `SESSION_SECRET`.

**VPS.** `docker-compose.yml` lance l'application et Caddy. Définir `DOMAIN` avant `docker compose up`.

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `app.py` | Site, comptes, socket web, flux et démo |
| `session.py` | Une surveillance par utilisateur connecté |
| `pipeline.py` | Fond, masque, objets |
| `zones.py` | Polygones |
| `faces.py` | Cascade de Haar |
| `alerts.py` | JPEG et CSV |
| `accounts.py` | SQLite et cookie |
| `main.py` | Démonstration en fenêtres OpenCV |
| `static/` | Pages de connexion et de surveillance |
