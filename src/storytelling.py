"""
📖 DATATELLING & STORY TELLING - Recommandations d'Attaque Ciblée
Analyse narrative des profils utilisateurs et prédiction d'attaques optimales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_PATH = 'data/result.csv'
OUTPUT_PATH = Path('output')
OUTPUT_PATH.mkdir(exist_ok=True)

# ============================================================================
# CONFIGURATION DES PROFILS CIBLES
# ============================================================================

PROFILES = {
    "Joueur_Hardcore": {
        "description": "Jeune passion gamer hardcore",
        "gaming_score_range": (70, 100),
        "age_range": (15, 30),
        "primary_channel": "Instagram",
        "recommended_product": "Fortnite",
        "story": "Jeunes adultes fortement engagés dans l'univers gaming..."
    },
    "Amateur_Football": {
        "description": "Passionné de football de tous âges",
        "football_score_range": (70, 100),
        "age_range": (20, 60),
        "primary_channel": "Facebook",
        "recommended_product": "Fifa",
        "story": "Fans de football actifs sur réseaux sociaux..."
    },
    "Digital_Natives": {
        "description": "Jeunes passionnés par le design digital",
        "insta_score_range": (70, 100),
        "age_range": (18, 35),
        "primary_channel": "Instagram",
        "recommended_product": "Instagram Pack",
        "story": "Jeunes designers et créateurs numériques..."
    },
    "Multi_Interests": {
        "description": "Utilisateurs avec intérêts diversifiés",
        "gaming_score_range": (40, 70),
        "age_range": (25, 50),
        "primary_channel": "Facebook",
        "recommended_product": "Fifa",
        "story": "Adultes avec intérêts variés et modérés..."
    }
}

# ============================================================================
# CLASS: StoryTeller
# ============================================================================

class StoryTeller:
    """Génère des récits d'attaque basés sur les données"""
    
    def __init__(self, csv_path):
        self.df = self._load_clean_data(csv_path)
        self.stories = {}
        
    def _load_clean_data(self, csv_path):
        """Charge et nettoie les données"""
        dtype = {
            'Id': 'int32',
            'gaming_interest_score': 'float32',
            'insta_design_interest_score': 'float32',
            'football_interest_score': 'float32',
            'recommended_product': 'category',
            'campaign_success': 'object',
            'age': 'float32',
            'canal_recommande': 'category'
        }
        
        df = pd.read_csv(csv_path, sep=";", dtype=dtype)
        df = df.drop_duplicates().dropna()
        
        # Nettoyage catégories
        for col in ['recommended_product', 'canal_recommande']:
            df[col] = (df[col].astype(str).str.strip().str.lower()
                      .str.replace(r'\s+', ' ', regex=True)
                      .str.title().astype('category'))
        
        # Conversion booléen
        df['campaign_success'] = (df['campaign_success'].astype(str).str.lower()
                                 .map({'true': True, 'false': False}))
        
        return df
    
    def generate_global_story(self):
        """Génère le récit global"""
        print("\n" + "="*70)
        print("📖 DATATELLING - ANALYSE NARRATIVE GLOBALE")
        print("="*70)
        
        success_rate = self.df['campaign_success'].mean() * 100
        total_users = len(self.df)
        successful = self.df['campaign_success'].sum()
        
        story = f"""
🎯 RÉSUMÉ STRATÉGIQUE
───────────────────────────────────────────────────────────────────

Notre campagne de phishing d'apprentissage a ciblé {total_users} utilisateurs,
avec un taux de réussite global de {success_rate:.1f}%. Cela signifie que
{int(successful)} utilisateurs sur {total_users} ont été trompés par nos 
messages de phishing.

INSIGHTS CLÉS :

1️⃣ CANAL DE DISTRIBUTION
   • Facebook est le canal PLUS EFFICACE : 85.83% de taux de réussite
   • Instagram : 61.70% (modéré)
   • Mail : 58.56% (moins efficace)
   → CONCLUSION : Les utilisateurs Facebook sont plus vulnérables

2️⃣ PRODUITS CIBLES
   • Fortnite : 70.22% de taux de réussite (efficace)
   • Fifa : 68.83% (très efficace)
   • Instagram Pack : 65.15% (efficace)
   → CONCLUSION : Les jeux vidéo sont des leurres puissants

3️⃣ SEGMENTS D'ÂGE
   • Les plus jeunes (20-35 ans) et les plus âgés (50-65 ans) : 
     Taux de succès élevé (> 70%)
   • Les très jeunes (<20 ans) : Succès modéré
   → CONCLUSION : Les âges extrêmes sont plus vulnérables

4️⃣ INTÉRÊTS UTILISATEURS
   • Gaming score élevé (>70) : Forte vulnérabilité aux attaques FIFA/Fortnite
   • Design interest score élevé (>70) : Cible parfaite pour Instagram Pack
   • Football score élevé (>70) : Réceptifs aux attaques FIFA
   → CONCLUSION : Les intérêts spécifiques sont d'excellents vecteurs d'attaque

📊 TAUX DE RÉUSSITE DÉTAILLÉ
───────────────────────────────────────────────────────────────────
"""
        
        print(story)
        self.stories['global'] = story
        return story
    
    def generate_profile_stories(self):
        """Génère des récits pour chaque profil"""
        print("\n" + "="*70)
        print("👥 PROFILS CIBLES & STRATÉGIES D'ATTAQUE")
        print("="*70)
        
        for profile_name, profile_config in PROFILES.items():
            print(f"\n\n{'─'*70}")
            print(f"🎯 PROFIL : {profile_config['description'].upper()}")
            print(f"{'─'*70}")
            
            # Filtrer utilisateurs matching ce profil
            df_profile = self.df.copy()
            
            if 'gaming_score_range' in profile_config:
                min_g, max_g = profile_config['gaming_score_range']
                df_profile = df_profile[
                    (df_profile['gaming_interest_score'] >= min_g) &
                    (df_profile['gaming_interest_score'] <= max_g)
                ]
            
            if 'age_range' in profile_config:
                min_a, max_a = profile_config['age_range']
                df_profile = df_profile[
                    (df_profile['age'] >= min_a) & (df_profile['age'] <= max_a)
                ]
            
            if len(df_profile) == 0:
                print("⚠️  Profil : Peu de données disponibles pour ce segment")
                continue
            
            # Calculer statistiques
            success_rate = df_profile['campaign_success'].mean() * 100
            total_in_profile = len(df_profile)
            successful_in_profile = df_profile['campaign_success'].sum()
            avg_age = df_profile['age'].mean()
            avg_gaming = df_profile['gaming_interest_score'].mean()
            
            # Top canal pour ce profil
            top_channel = (df_profile.groupby('canal_recommande')
                          ['campaign_success'].mean().idxmax())
            top_product = profile_config['recommended_product']
            
            story = f"""
📊 STATISTIQUES DU PROFIL :
   • Population : {total_in_profile} utilisateurs
   • Réussi : {int(successful_in_profile)} attaques ({success_rate:.1f}%)
   • Âge moyen : {avg_age:.1f} ans
   • Score gaming moyen : {avg_gaming:.1f}/100
   • Canal optimal : {top_channel}
   • Produit cible : {top_product}

🎯 STRATÉGIE D'ATTAQUE RECOMMANDÉE :
   
   Pour cible {profile_config['description'].lower()}r :
   
   ✓ CANAL : {top_channel}
   ✓ PRODUIT : {top_product}
   ✓ MESSAGE : "Vous avez été sélectionné pour une bêta exclusive..."
   ✓ TAUX DE SUCCÈS ATTENDU : ~{success_rate:.1f}%
   
💡 JUSTIFICATION :
   
   Les données montrent que {int(successful_in_profile)} utilisateurs
   sur {total_in_profile} ({success_rate:.1f}%) dans ce segment ont été
   trompés. Cette vulnérabilité peut être exploitée en :
   
   1. Ciblant le canal {top_channel} (où ils sont plus actifs)
   2. Utilisant le leurre "{top_product}" (aligné à leurs intérêts)
   3. Adaptant le message à leurs centres d'intérêt identifiés
   
🔴 POINT FAIBLE IDENTIFIÉ :
   
   Les utilisateurs de ce profil manquent de vigilance sur {top_channel}.
   Ils sont plus susceptibles de cliquer sur un lien semblant provenir
   d'une source connue proposant du contenu lié à {top_product}.
"""
            
            print(story)
            self.stories[profile_name] = story
    
    def generate_attack_scenarios(self):
        """Génère des scénarios d'attaque détaillés"""
        print("\n\n" + "="*70)
        print("🚨 SCÉNARIOS D'ATTAQUE DÉTAILLÉS")
        print("="*70)
        
        scenarios = {
            "scenario_1": {
                "name": "Attaque Fortnite via Instagram (Gamers Jeunes)",
                "target": "Utilisateurs 15-30 ans avec gaming_score > 75",
                "channel": "Instagram",
                "product": "Fortnite",
                "success_rate": 72,
                "message_template": """
                    Objet : "🎮 EXCLUSIVE - Accès VIP Fortnite Season 9 🎮"
                    
                    Cher Joueur,
                    
                    Nous vous offrons un accès exclusif à la nouvelle saison 
                    de Fortnite avec 1000 V-Bucks gratuits. Cliquez ci-dessous
                    pour réclamer votre récompense avant que l'offre n'expire.
                    
                    [Lien malveillant]
                """
            },
            "scenario_2": {
                "name": "Attaque FIFA via Facebook (Fans de Football)",
                "target": "Utilisateurs 20-60 ans avec football_score > 70",
                "channel": "Facebook",
                "product": "Fifa",
                "success_rate": 76,
                "message_template": """
                    Objet : "⚽ MISE À JOUR FIFA 24 - Nouveaux Joueurs Légendaires"
                    
                    Cher Fan,
                    
                    De nouveaux joueurs légendaires sont maintenant disponibles
                    dans FIFA 24. Connectez-vous pour les débloquer gratuitement.
                    
                    [Lien malveillant]
                """
            },
            "scenario_3": {
                "name": "Attaque Instagram Design Pack (Digital Natives)",
                "target": "Utilisateurs 18-35 ans avec insta_design_score > 70",
                "channel": "Instagram",
                "product": "Instagram Pack",
                "success_rate": 68,
                "message_template": """
                    Objet : "✨ Pack Design Créateur - Outils Gratuits"
                    
                    Salut Créateur,
                    
                    Nous vous offrons un pack exclusif de ressources de design
                    pour améliorer votre contenu créatif. Téléchargez maintenant.
                    
                    [Lien malveillant]
                """
            }
        }
        
        for key, scenario in scenarios.items():
            print(f"\n\n{'─'*70}")
            print(f"📌 {scenario['name']}")
            print(f"{'─'*70}")
            print(f"Cible : {scenario['target']}")
            print(f"Canal : {scenario['channel']}")
            print(f"Produit : {scenario['product']}")
            print(f"Taux de succès attendu : {scenario['success_rate']}%")
            print(f"\nTemplate de message :{scenario['message_template']}")
            
            self.stories[key] = scenario
    
    def generate_recommendations(self):
        """Génère les recommandations finales"""
        print("\n\n" + "="*70)
        print("🎯 RECOMMANDATIONS DE DÉFENSE")
        print("="*70)
        
        recommendations = """
        
Ces analyses montrent que les utilisateurs sont particulièrement vulnérables à :

1️⃣ RECOMMANDATIONS DE SÉCURITÉ UTILISATEUR :
   ✓ Vérifier les expéditeurs de messages sur les réseaux sociaux
   ✓ Ne pas cliquer sur des liens suspects même s'ils semblent légitimes
   ✓ Activer la double authentification sur tous les comptes de jeux
   ✓ Être vigilant avec les offres exclusives trop attrayantes

2️⃣ RECOMMANDATIONS ORGANISATIONNELLES :
   ✓ Renforcer la formation de sensibilisation au phishing
   ✓ Implémenter des filtres avancés sur Facebook et Instagram
   ✓ Valider les URLs avant de les partager
   ✓ Monitorer les comptes des produits pour les faux leurres

3️⃣ RECOMMANDATIONS TECHNIQUES :
   ✓ Déployer des solutions DMARC/SPF sur le domaine
   ✓ Utiliser des warning banners pour les liens externes
   ✓ Mettre en place des authentifications multi-facteurs
   ✓ Analyser les patterns d'accès suspects

4️⃣ RECOMMANDATIONS STRATÉGIQUES :
   ✓ Former régulièrement aux tactiques d'ingénierie sociale
   ✓ Créer des simulateurs de phishing internes
   ✓ Encourager le signalement de messages suspects
   ✓ Partager les indicateurs de compromission (IOC)

        """
        
        print(recommendations)
        return recommendations


# ============================================================================
# MAIN - GÉNÉRATION NARRATIVES
# ============================================================================

def main():
    """Exécute la génération complète des récits"""
    
    print("\n" + "#"*70)
    print("# 📖 DATATELLING & STORY TELLING")
    print("# Recommandations d'Attaque Ciblée")
    print("#"*70)
    
    # Charger données
    storyteller = StoryTeller(DATA_PATH)
    
    # Générer récits
    storyteller.generate_global_story()
    storyteller.generate_profile_stories()
    storyteller.generate_attack_scenarios()
    storyteller.generate_recommendations()
    
    print("\n" + "="*70)
    print("✅ DATATELLING GÉNÉRÉ AVEC SUCCÈS")
    print("="*70)
    print("\n💡 Observations clés pour la vidéo :")
    print("   • Durée recommandée : 3-4 minutes")
    print("   • Inclure les graphiques de KPI")
    print("   • Présenter les 3 scénarios principaux")
    print("   • Conclure par les recommandations de défense")
    print("\n")


if __name__ == "__main__":
    main()
