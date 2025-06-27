# CoreConstruct

**CoreConstruct** is an AI-powered system that automates the generation of architectural foundation plans using image recognition and deep reinforcement learning. It utilizes **YOLOv8** for identifying structural features from floor plans and a **Variational Autoencoder (VAE)**-based **Reinforcement Learning (RL)** model to generate efficient, code-compliant foundation layouts.

---

## 📄 Abstract

Manual foundation plan creation is time-consuming and prone to human error. CoreConstruct bridges AI and architectural design by:

- Detecting architectural elements from input images
- Generating foundation plans through AI-based decision making
- Ensuring accuracy, adherence to codes, and efficient computation

**Key results:**
- 🧱 Column Size Accuracy: **98.54%**
- 📏 Footing Size Accuracy: **93.21%**
- 📊 Overall Accuracy: **70.9%**
- ⚙️ Precision: **98.86%**, Recall: **77.93%**, F1 Score: **87.15%**
- ⚡ Processing Times: **0.08s (RL)**, **14.76s (VAE)**

---

## 🚀 Features

- 🧠 **YOLOv8**-based image recognition for floor plans
- 🔁 **VAE + Reinforcement Learning** for foundation optimization
- ✅ Compliance verification with Philippine building codes
- 🛠 Customizable inputs (e.g., soil type, material, storey count)
- 🧾 Annotated output includes loads, depth, size, reinforcement bars, etc.

---

## ⚙️ Technologies Used

- Python
- YOLOv8 (Ultralytics)
- TensorFlow / PyTorch
- OpenCV
- NumPy, SciPy
- JSON for configuration
- `.cct` proprietary CoreConstruct file format

---

## 📊 Evaluation Metrics

- **Plan Accuracy**: Size & placement of footings/columns
- **Compliance**: Adherence to NSCP & local codes
- **Plan Completeness**: Detail levels in outputs
- **Efficiency**: Time to generate optimized plans

---

## 🚫 Limitations

- Supports only **1–2 storey residential/commercial** floor plans
- Slanted/curved wall recognition not fully supported
- Focused on **top-view layouts only**
- Outputs image-based files (PNG, PDF, `.cct`) only

---

## 👥 Authors

- **Joshua P. Dale**
- **Hershey Ann C. Estoya**
- **Shun Buk P. Francisco**
- **Kyle R. Magnaye**
- **Rafael M. Quipit**

Developed as part of a thesis for the **Polytechnic University of the Philippines**  
**College of Computer and Information Sciences** (2024–2025)

---

## 📜 License

© 2024 by the Authors and Polytechnic University of the Philippines.  
Portions may be reused with proper citation and acknowledgment.

---

## 📫 Contact

For academic or development inquiries, please open an [issue](https://github.com/) or fork the project.

