"""
📊 Campaign Data Analysis Pipeline
Analyse complète des données de campagne marketing avec détection d'outliers,
nettoyage et génération de KPI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

NUMERIC_COLUMNS = [
    'gaming_interest_score',
    'insta_design_interest_score',
    'football_interest_score',
    'age'
]

CATEGORICAL_COLUMNS = [
    'recommended_product',
    'canal_recommande'
]

COLUMN_DTYPES = {
    'Id': 'int32',
    'gaming_interest_score': 'float32',
    'insta_design_interest_score': 'float32',
    'football_interest_score': 'float32',
    'recommended_product': 'category',
    'campaign_success': 'object',
    'age': 'float32',
    'canal_recommande': 'category'
}

# ============================================================================
# CLASS: DataProcessor
# ============================================================================

class DataProcessor:
    """Gère le chargement, nettoyage et transformation des données"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.df_original = None
        
    def load_data(self):
        """Charge le CSV avec types optimisés"""
        print("\n📥 CHARGEMENT DES DONNÉES")
        print("-" * 60)
        
        self.df = pd.read_csv(self.csv_path, sep=";", dtype=COLUMN_DTYPES)
        self.df_original = self.df.copy()
        
        print(f"✓ Dataset chargé : {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")
        print(f"\n📋 Aperçu du dataset :\n{self.df.head()}")
        print(f"\n📊 Types de données :\n{self.df.dtypes}")
        
        return self
    
    def check_memory(self):
        """Affiche l'utilisation mémoire avant/après"""
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024**2
        print(f"\n💾 Mémoire utilisée : {memory_mb:.3f} Mo")
        return memory_mb
    
    def remove_duplicates(self):
        """Supprime les enregistrements dupliqués"""
        nb_before = len(self.df)
        self.df = self.df.drop_duplicates()
        nb_removed = nb_before - len(self.df)
        
        if nb_removed > 0:
            print(f"🗑️  {nb_removed} doublon(s) supprimé(s)")
        else:
            print(f"✓ Aucun doublon détecté")
        
        return self
    
    def handle_missing_values(self):
        """Traite les valeurs manquantes"""
        missing = self.df.isna().sum()
        
        if missing.sum() > 0:
            print(f"\n❌ Valeurs manquantes détectées :\n{missing[missing > 0]}")
            self.df = self.df.dropna()
            print(f"✓ Lignes avec NA supprimées. Dataset : {len(self.df)} lignes")
        else:
            print(f"✓ Aucune valeur manquante")
        
        return self
    
    def clean_categorical(self):
        """Normalise les colonnes catégorielles"""
        print(f"\n🔤 NETTOYAGE DES CATÉGORIES")
        print("-" * 60)
        
        for col in CATEGORICAL_COLUMNS:
            self.df[col] = (self.df[col]
                           .astype(str)
                           .str.strip()
                           .str.lower()
                           .str.replace(r'\s+', ' ', regex=True)
                           .str.title()
                           .astype('category'))
        
        print(f"✓ Catégories standardisées")
        
        return self
    
    def clean_boolean(self):
        """Nettoie la colonne campaign_success"""
        self.df['campaign_success'] = (self.df['campaign_success']
                                       .astype(str)
                                       .str.strip()
                                       .str.lower()
                                       .map({'true': True, 'false': False}))
        
        print(f"✓ Colonne booléenne convertie")
        return self
    
    def get_cleaned_data(self):
        """Retourne le dataframe nettoyé"""
        return self.df


# ============================================================================
# CLASS: OutlierDetector
# ============================================================================

class OutlierDetector:
    """Détecte et gère les valeurs extrêmes"""
    
    def __init__(self, dataframe):
        self.df = dataframe
        self.outliers_mask = None
        self.outliers_info = {}
    
    def detect_iqr(self):
        """Détecte les outliers avec la méthode IQR"""
        print(f"\n🔍 DÉTECTION DES ANOMALIES (IQR)")
        print("-" * 60)
        
        self.outliers_mask = pd.DataFrame(False, index=self.df.index, columns=NUMERIC_COLUMNS)
        
        for col in NUMERIC_COLUMNS:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            self.outliers_mask[col] = outliers
            
            self.outliers_info[col] = {
                'count': outliers.sum(),
                'lower': lower_bound,
                'upper': upper_bound
            }
            
            print(f"\n{col}:")
            print(f"  Bounds : [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  Anomalies : {outliers.sum()}")
        
        total_outliers = self.outliers_mask.any(axis=1).sum()
        print(f"\n📊 Total lignes avec anomalies : {total_outliers}")
        
        return self
    
    def visualize_outliers(self, save_path='output/'):
        """Crée des boxplots pour chaque colonne numérique"""
        print(f"\n📈 VISUALISATION DES OUTLIERS")
        print("-" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(NUMERIC_COLUMNS):
            axes[idx].boxplot(self.df[col], vert=True)
            axes[idx].set_title(f"Distribution : {col}", fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(col)
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}01_outliers_boxplots.png', dpi=100, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé : 01_outliers_boxplots.png")
        plt.show()
        
        return self
    
    def get_clean_data(self):
        """Retourne le dataset sans outliers"""
        return self.df[~self.outliers_mask.any(axis=1)].copy()


# ============================================================================
# CLASS: KPIAnalyzer
# ============================================================================

class KPIAnalyzer:
    """Calcule les indicateurs clés de performance"""
    
    def __init__(self, dataframe):
        self.df = dataframe
        self.kpis = {}
    
    def calculate_global_rate(self):
        """Taux de réussite global"""
        rate = self.df['campaign_success'].mean()
        self.kpis['global_success_rate'] = rate
        
        print(f"\n✅ Taux de réussite global : {rate*100:.2f}%")
        
        return self
    
    def calculate_by_product(self):
        """Taux par produit recommandé"""
        rates = self.df.groupby('recommended_product')['campaign_success'].agg([
            ('taux', 'mean'),
            ('count', 'size')
        ]).round(4)
        
        self.kpis['by_product'] = rates
        
        print(f"\n📦 Taux de réussite par produit :")
        print(rates)
        
        return self
    
    def calculate_by_channel(self):
        """Taux par canal de recommandation"""
        rates = self.df.groupby('canal_recommande')['campaign_success'].agg([
            ('taux', 'mean'),
            ('count', 'size')
        ]).round(4)
        
        self.kpis['by_channel'] = rates
        
        print(f"\n📡 Taux de réussite par canal :")
        print(rates)
        
        return self
    
    def calculate_by_age_group(self):
        """Taux par groupe d'âge"""
        df_temp = self.df.copy()
        df_temp['age_group'] = pd.cut(
            df_temp['age'],
            bins=[0, 20, 35, 50, 65, 120],
            labels=['<20', '20-35', '35-50', '50-65', '65+']
        )
        
        rates = df_temp.groupby('age_group', observed=True)['campaign_success'].agg([
            ('taux', 'mean'),
            ('count', 'size')
        ]).round(4)
        
        self.kpis['by_age_group'] = rates
        
        print(f"\n👥 Taux de réussite par groupe d'âge :")
        print(rates)
        
        return self
    
    def visualize_kpis(self, save_path='output/'):
        """Crée des graphiques pour les KPI"""
        print(f"\n📊 VISUALISATION DES KPI")
        print("-" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Global rate
        global_rate = self.kpis['global_success_rate']
        axes[0, 0].bar(['Succès', 'Échec'], [global_rate, 1-global_rate], 
                       color=['#2ecc71', '#e74c3c'])
        axes[0, 0].set_title('Taux Global de Réussite', fontweight='bold')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # By product
        self.kpis['by_product']['taux'].plot(kind='barh', ax=axes[0, 1], color='skyblue')
        axes[0, 1].set_title('Taux par Produit', fontweight='bold')
        axes[0, 1].set_xlabel('Taux de réussite')
        axes[0, 1].set_xlim([0, 1])
        
        # By channel
        self.kpis['by_channel']['taux'].plot(kind='bar', ax=axes[1, 0], color='lightcoral')
        axes[1, 0].set_title('Taux par Canal', fontweight='bold')
        axes[1, 0].set_ylabel('Taux de réussite')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # By age group
        self.kpis['by_age_group']['taux'].plot(kind='line', ax=axes[1, 1], 
                                               marker='o', linewidth=2, markersize=8, color='#9b59b6')
        axes[1, 1].set_title('Taux par Groupe d\'Âge', fontweight='bold')
        axes[1, 1].set_ylabel('Taux de réussite')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}02_kpi_analysis.png', dpi=100, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé : 02_kpi_analysis.png")
        plt.show()
        
        return self


# ============================================================================
# CLASS: CorrelationAnalyzer
# ============================================================================

class CorrelationAnalyzer:
    """Analyse les corrélations entre variables"""
    
    def __init__(self, dataframe):
        self.df = dataframe
        self.correlation_matrix = None
    
    def compute_correlation(self):
        """Calcule la matrice de corrélation"""
        print(f"\n🔗 ANALYSE DE CORRÉLATION")
        print("-" * 60)
        
        # Convertir campaign_success en numérique
        df_temp = self.df.copy()
        df_temp['campaign_success'] = df_temp['campaign_success'].astype(int)
        
        # Sélectionner colonnes numériques
        numeric_data = df_temp[NUMERIC_COLUMNS + ['campaign_success']]
        self.correlation_matrix = numeric_data.corr()
        
        print("\n📊 Matrice de corrélation :")
        print(self.correlation_matrix.round(3))
        
        # Top corrélations avec campaign_success
        correlations = self.correlation_matrix['campaign_success'].sort_values(ascending=False)
        print(f"\n⭐ Corrélations avec le succès :")
        for col, val in correlations.items():
            if col != 'campaign_success':
                print(f"  {col}: {val:.3f}")
        
        return self
    
    def visualize_correlation(self, save_path='output/'):
        """Crée une heatmap de corrélation"""
        print(f"\n📈 VISUALISATION DE CORRÉLATION")
        print("-" * 60)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.correlation_matrix, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, 
                   cbar_kws={'label': 'Corrélation'})
        plt.title('Matrice de Corrélation - Variables Numériques', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{save_path}03_correlation_heatmap.png', dpi=100, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé : 03_correlation_heatmap.png")
        plt.show()
        
        return self


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Exécute le pipeline d'analyse complet"""
    
    print("\n" + "="*60)
    print("🚀 CAMPAIGN DATA ANALYSIS PIPELINE")
    print("="*60)
    
    # ÉTAPE 1: Charger et nettoyer les données
    processor = DataProcessor('data/result.csv')
    processor.load_data()
    processor.check_memory()
    
    print(f"\n🧹 NETTOYAGE DES DONNÉES")
    print("-" * 60)
    processor.remove_duplicates()
    processor.handle_missing_values()
    processor.clean_categorical()
    processor.clean_boolean()
    processor.check_memory()
    
    df_clean = processor.get_cleaned_data()
    
    # ÉTAPE 2: Détecter les outliers
    detector = OutlierDetector(df_clean)
    detector.detect_iqr()
    detector.visualize_outliers()
    
    df_no_outliers = detector.get_clean_data()
    print(f"\n✓ Dataset après suppression outliers : {len(df_no_outliers)} lignes")
    
    # ÉTAPE 3: Calculer les KPI
    print(f"\n📊 ANALYSE DES KPI")
    print("-" * 60)
    analyzer = KPIAnalyzer(df_no_outliers)
    analyzer.calculate_global_rate()
    analyzer.calculate_by_product()
    analyzer.calculate_by_channel()
    analyzer.calculate_by_age_group()
    analyzer.visualize_kpis()
    
    # ÉTAPE 4: Analyser les corrélations
    corr_analyzer = CorrelationAnalyzer(df_no_outliers)
    corr_analyzer.compute_correlation()
    corr_analyzer.visualize_correlation()
    
    # Résumé final
    print(f"\n" + "="*60)
    print("✅ ANALYSE COMPLÉTÉE AVEC SUCCÈS")
    print("="*60)
    print(f"\n📋 Résumé :")
    print(f"  • Dataset initial : {len(processor.df_original)} lignes")
    print(f"  • Dataset après nettoyage : {len(df_clean)} lignes")
    print(f"  • Dataset final (sans outliers) : {len(df_no_outliers)} lignes")
    print(f"  • Graphiques générés dans : output/")
    print("\n")


if __name__ == "__main__":
    main()
