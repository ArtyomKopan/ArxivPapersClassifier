import streamlit as st
import random

# Таксономия arXiv (основные категории)
ARXIV_CATEGORIES = [
    "cs.AI - Artificial Intelligence",
    "cs.CL - Computation and Language",
    "cs.CV - Computer Vision and Pattern Recognition",
    "cs.LG - Machine Learning",
    "cs.NE - Neural and Evolutionary Computing",
    "cs.RO - Robotics",
    "cs.CR - Cryptography and Security",
    "cs.DB - Databases",
    "cs.DC - Distributed, Parallel, and Cluster Computing",
    "cs.DS - Data Structures and Algorithms",
    "cs.GT - Computer Science and Game Theory",
    "cs.IR - Information Retrieval",
    "cs.IT - Information Theory",
    "cs.LO - Logic in Computer Science",
    "cs.MA - Multiagent Systems",
    "cs.NI - Networking and Internet Architecture",
    "cs.OS - Operating Systems",
    "cs.PL - Programming Languages",
    "cs.SD - Sound",
    "cs.SE - Software Engineering",
    "math.AC - Commutative Algebra",
    "math.AG - Algebraic Geometry",
    "math.AP - Analysis of PDEs",
    "math.AT - Algebraic Topology",
    "math.CA - Classical Analysis",
    "math.CO - Combinatorics",
    "math.CT - Category Theory",
    "math.CV - Complex Variables",
    "math.DG - Differential Geometry",
    "math.DS - Dynamical Systems",
    "math.FA - Functional Analysis",
    "math.GM - General Mathematics",
    "math.GN - General Topology",
    "math.GR - Group Theory",
    "math.HO - History and Overview",
    "math.KT - K-Theory and Homology",
    "math.LO - Logic",
    "math.MP - Mathematical Physics",
    "math.NA - Numerical Analysis",
    "math.NT - Number Theory",
    "math.OA - Operator Algebras",
    "math.OC - Optimization and Control",
    "math.PR - Probability",
    "math.QA - Quantum Algebra",
    "math.RA - Rings and Algebras",
    "math.RT - Representation Theory",
    "math.SG - Symplectic Geometry",
    "math.SP - Spectral Theory",
    "math.ST - Statistics Theory",
    "physics.optics - Optics",
    "physics.bio-ph - Biological Physics",
    "physics.chem-ph - Chemical Physics",
    "physics.class-ph - Classical Physics",
    "physics.comp-ph - Computational Physics",
    "physics.data-an - Data Analysis",
    "physics.ed-ph - Physics Education",
    "physics.flu-dyn - Fluid Dynamics",
    "physics.gen-ph - General Physics",
    "physics.geo-ph - Geophysics",
    "physics.hist-ph - History of Physics",
    "physics.ins-det - Instrumentation and Detectors",
    "physics.med-ph - Medical Physics",
    "physics.plasm-ph - Plasma Physics",
    "physics.pop-ph - Popular Physics",
    "physics.soc-ph - Physics and Society",
    "physics.space-ph - Space Physics",
    "q-bio.BM - Biomolecules",
    "q-bio.CB - Cell Behavior",
    "q-bio.GN - Genomics",
    "q-bio.MN - Molecular Networks",
    "q-bio.NC - Neurons and Cognition",
    "q-bio.OT - Other Quantitative Biology",
    "q-bio.PE - Populations and Evolution",
    "q-bio.QM - Quantitative Methods",
    "q-bio.SC - Subcellular Processes",
    "q-bio.TO - Tissues and Organs",
    "q-fin.CP - Computational Finance",
    "q-fin.GN - General Finance",
    "q-fin.MF - Mathematical Finance",
    "q-fin.PM - Portfolio Management",
    "q-fin.PR - Pricing of Securities",
    "q-fin.RM - Risk Management",
    "q-fin.ST - Statistical Finance",
    "q-fin.TR - Trading and Market Microstructure",
    "stat.AP - Applications",
    "stat.CO - Computation",
    "stat.ME - Methodology",
    "stat.ML - Machine Learning",
    "stat.TH - Statistics Theory"
]

def main():
    # Настройка страницы
    st.set_page_config(
        page_title="arXiv Papers Classifier",
        page_icon="📚",
        layout="wide"
    )
    
    # Заголовок приложения
    st.title("📚 arXiv papers classifier")
    
    # Создание двух колонок для формы ввода и результатов
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Форма ввода заголовка
        title = st.text_area(
            "**Input title:**",
            placeholder="Enter the paper title here...",
            height=100
        )
        
        # Форма ввода аннотации (больше по высоте)
        abstract = st.text_area(
            "**Input abstract:**",
            placeholder="Enter the abstract here...",
            height=300
        )
        
        # Кнопка для классификации
        classify_button = st.button("🔍 Classify Paper", type="primary", use_container_width=True)
    
    with col2:
        # Поле для вывода результатов
        st.markdown("**The most probable categories**")
        
        # Создание контейнера для результатов
        results_container = st.container(border=True)
        
        with results_container:
            if classify_button:
                # Проверка, что введен хотя бы заголовок или аннотация
                if title.strip() or abstract.strip():
                    # Генерация 5 случайных категорий
                    random_categories = random.sample(ARXIV_CATEGORIES, 5)
                    
                    # Отображение результатов
                    for i, category in enumerate(random_categories, 1):
                        st.markdown(f"**{i}.** `{category}`")
                    
                    # Дополнительная информация
                    st.divider()
                    st.caption("⚠️ This is a demo version. Categories are randomly generated.")
                else:
                    st.warning("Please enter a title or abstract to classify.")
            else:
                # Заглушка до нажатия кнопки
                st.info("👈 Enter paper details and click 'Classify Paper' to see results")
                st.caption("The classifier will display 5 random arXiv categories as a demo.")
    
    # Дополнительная информация внизу страницы
    with st.expander("ℹ️ About this app"):
        st.markdown("""
        This is a **demo application** for arXiv papers classification.
        
        **Features:**
        - Input paper title and abstract
        - Returns 5 random categories from the arXiv taxonomy
        
        **Next steps:**
        - Replace random selection with actual ML model
        - Implement real classification logic
        - Add confidence scores for predictions
        
        **arXiv taxonomy includes categories from:**
        - Computer Science (cs.*)
        - Mathematics (math.*)
        - Physics (physics.*)
        - Quantitative Biology (q-bio.*)
        - Quantitative Finance (q-fin.*)
        - Statistics (stat.*)
        """)

if __name__ == "__main__":
    main()
