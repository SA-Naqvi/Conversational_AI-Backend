"""
Corpus generator for the Medical Recovery Companion RAG system.
Generates 60 realistic medical documents covering post-surgery recovery topics.
Run this script once before building the vector index.
"""
import os
import pathlib

DOCS_DIR = pathlib.Path(__file__).parent.parent / "data" / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

DOCUMENTS = {
    "doc_01_general_post_op_care.txt": """
GENERAL POST-OPERATIVE CARE INSTRUCTIONS

Overview
Following surgery, proper post-operative care is critical to ensure a smooth and timely recovery. These instructions apply broadly to most surgical procedures and should be supplemented by any specific guidance provided by your surgeon.

Immediate Post-Surgery (First 24–48 Hours)
Upon returning home, rest is the top priority. Keep the surgical site elevated when possible to reduce swelling. Avoid bending or twisting movements that stress the incision area. Do not drive for at least 24 hours after receiving general or regional anesthesia, as reaction time and judgment may be impaired.

You may feel drowsy, nauseous, or dizzy following anesthesia — this is normal and typically resolves within 24 hours. Eat light foods such as crackers, toast, or broth if nausea is present. Gradually return to a regular diet as tolerated.

Pain Management
Mild to moderate pain is expected after surgery. Take prescribed pain medications as directed by your healthcare provider. Do not wait until pain becomes severe before taking medication — staying ahead of pain is more effective. Avoid alcohol while taking prescription pain medications. If over-the-counter medications like acetaminophen or ibuprofen were recommended, follow dosage instructions carefully.

Activity Restrictions
Avoid strenuous activities, heavy lifting (typically nothing over 10 lbs for the first two weeks), or vigorous exercise until cleared by your surgeon. Light walking is encouraged starting the day after surgery to promote circulation and reduce the risk of blood clots.

Follow-Up Appointments
Attend all scheduled post-operative appointments. These visits allow your surgeon to monitor healing, remove sutures or staples if needed, and address any concerns. Missing these appointments can compromise your recovery outcome.

When to Call Your Doctor
Contact your healthcare provider immediately if you experience: fever above 101.5°F (38.6°C), increasing redness, warmth, or discharge from the incision, severe or worsening pain not relieved by medication, chest pain or difficulty breathing, swelling or redness in a leg (potential blood clot), or if you cannot keep fluids down.

Medications and Supplements
Continue taking all regularly prescribed medications unless instructed otherwise. Inform your doctor about any supplements, vitamins, or herbal products you take, as some can interfere with healing or interact with prescribed medications.
""",

    "doc_02_wound_care_instructions.txt": """
WOUND CARE AFTER SURGERY

Understanding Your Surgical Wound
A surgical incision is a controlled wound that your body heals through a natural process involving inflammation, tissue formation, and remodeling. Proper wound care significantly reduces the risk of infection and promotes faster, cleaner healing.

Keeping the Wound Clean
For the first 24 to 48 hours, keep the wound dry and covered with the original dressing unless your surgeon instructs otherwise. After this period, most surgeons recommend gently cleaning the wound with mild soap and water. Pat dry with a clean cloth. Do not scrub or rub the incision.

Dressing Changes
Change dressings as directed by your care team — typically once or twice daily. Wash your hands thoroughly before and after touching the wound. Use sterile gauze and medical tape; avoid adhesive bandages that do not allow the skin to breathe.

What Normal Healing Looks Like
In the first few days, some redness, mild swelling, and clear or slightly blood-tinged drainage are normal. Bruising around the incision is also common. Over 1–2 weeks, the wound edges should knit together, redness should fade, and any drainage should stop.

Signs of Infection
Seek medical attention if you notice: increasing (not decreasing) redness or warmth, thick yellow, green, or foul-smelling discharge, the wound edges separating (dehiscence), fever above 101.5°F, or increasing pain at the wound site after the first two days.

Protecting the Wound
Keep the wound out of pools, hot tubs, and bathtubs until fully healed (usually 2–4 weeks). Showers are generally permitted once the wound is sealed (no open areas). Protect healing wounds from direct sunlight to minimize scarring — use SPF 30+ sunscreen once fully healed and for up to one year after.

Steri-Strips and Staples
If your wound was closed with steri-strips (adhesive strips), let them fall off on their own — this typically takes 7–10 days. Do not pick at or remove them prematurely. Staples are usually removed by your healthcare provider at the 7–14 day follow-up appointment.

Scar Management
Once the wound is fully closed (no scabs, no open areas), you can begin scar management. Silicone gel sheets or silicone-based scar creams applied consistently for 3–6 months can significantly reduce scar visibility. Gentle massage of the scar area after it is fully healed can help reduce firmness.
""",

    "doc_03_pain_management_guide.txt": """
MANAGING PAIN AFTER SURGERY

Why Pain Occurs
Surgical pain results from tissue injury and the body's inflammatory response. Understanding that some pain is a normal part of healing helps patients manage expectations. However, uncontrolled pain can slow recovery, impair sleep, and reduce the ability to perform deep breathing and mobility exercises.

Pain Scales
Healthcare providers use pain scales (0–10) to assess and communicate pain levels. A score of 0 means no pain; 10 means the worst imaginable pain. The goal of pain management is to keep pain at a level where you can perform normal activities, deep breathing, and sleep — typically a score of 3 or below.

Multimodal Pain Management
Modern pain management uses multiple strategies simultaneously:
1. Prescription opioids (for moderate-severe pain): hydrocodone, oxycodone — take as prescribed, taper as pain decreases
2. Non-opioid analgesics: acetaminophen (Tylenol) — safe for most patients, take with food
3. Anti-inflammatory medications: ibuprofen (Advil), naproxen — reduce inflammation; take with food; check with surgeon first
4. Ice therapy: apply for 15–20 minutes every 1–2 hours for the first 48 hours to reduce swelling and pain
5. Positioning: elevation reduces swelling and associated pain

Opioid Safety
If you are prescribed opioid pain medication, take it only as directed. Never take more than prescribed. Do not combine with alcohol or other sedatives. Opioids cause constipation — use stool softeners proactively. Store opioids securely and dispose of unused medication safely at a pharmacy take-back program.

Non-Pharmacological Approaches
Beyond medications: deep breathing, gentle movement, distraction (music, podcasts, conversation), heat therapy after the first 72 hours, and relaxation techniques such as progressive muscle relaxation can all reduce pain perception.

Pain During Coughing or Deep Breathing
A technique called "splinting" helps: hold a pillow firmly against your abdomen or chest incision when coughing or taking deep breaths. This reduces pain from incision movement while still allowing effective lung expansion.

When Pain Requires Medical Attention
Contact your surgeon if: pain is not controlled with prescribed medications, pain suddenly worsens after initially improving, pain is accompanied by fever or unusual drainage, or you develop new symptoms such as chest pain, shortness of breath, or severe leg pain.
""",

    "doc_04_activity_restrictions.txt": """
ACTIVITY RESTRICTIONS AFTER SURGERY

The Importance of Gradual Return to Activity
After surgery, the body needs time to heal. Returning to activity too quickly increases risks including wound dehiscence, hernia formation, bleeding, and re-injury. Conversely, too little movement increases the risk of blood clots, pneumonia, and muscle weakness. Striking the right balance — guided by your surgical team — is essential.

General Restrictions in the First Week
Walking: Short, gentle walks are encouraged from day 1 or 2. Start with 5–10 minutes, two to three times daily. Avoid hills or uneven terrain. Lifting: Most patients are restricted to lifting nothing heavier than a gallon of milk (approximately 8 lbs) for the first 2 weeks. Bending/twisting: Avoid movements that stress the incision site. Driving: Do not drive while taking opioid medications or until you can react quickly in an emergency situation (typically 1–2 weeks minimum, depending on surgery type).

Weeks 2–6 Activity Progression
Gradual increase in walking distance and duration. Light household activities may be resumed as tolerated. Avoid activities that cause pain, significant fatigue, or incision stress. Sexual activity: Generally safe to resume at 3–6 weeks, once incision is healed and pain allows — confirm with your surgeon.

Returning to Work
Desk/sedentary jobs: 1–2 weeks post-surgery for minor procedures; 2–4 weeks for major abdominal or orthopedic surgery. Physical/manual labor: 4–8 weeks or longer; requires surgical clearance. Work from home if possible to reduce commute strain.

Exercise and Sports
Most sports and vigorous exercise are restricted for 4–8 weeks depending on surgery type. Your surgeon will guide you on when to resume. Start with low-impact activities (swimming, cycling) before high-impact (running, contact sports).

Warning Signs During Activity
Stop activity immediately and contact your surgeon if you experience: sudden sharp pain at the incision, wound opening, new or increasing swelling, shortness of breath, chest pain, or dizziness.
""",

    "doc_05_diet_after_surgery.txt": """
NUTRITION AND DIET AFTER SURGERY

Why Nutrition Matters for Recovery
Proper nutrition accelerates wound healing, supports immune function, rebuilds muscle mass lost during illness and surgery, and prevents constipation caused by opioid medications and reduced activity. Poor nutrition after surgery can prolong recovery and increase complication risks.

Immediate Post-Surgery Diet (Days 1–3)
If nausea or reduced appetite is present, start with clear liquids: water, broth, apple juice, gelatin. Progress to full liquids: milk, smoothies, pureed soups. Advance to soft foods: crackers, toast, scrambled eggs, yogurt, oatmeal, as tolerated.

Protein — The Building Block of Healing
Protein requirements increase significantly after surgery. Aim for 1.2–1.5 grams of protein per kilogram of body weight daily (e.g., 80–100g for a 70 kg adult). Good protein sources: eggs, fish, chicken, lean beef, Greek yogurt, cottage cheese, legumes, protein shakes.

Micronutrients for Healing
Vitamin C (50–200 mg/day): critical for collagen synthesis; found in citrus, bell peppers, strawberries. Zinc (8–11 mg/day): accelerates wound healing; found in meat, seeds, nuts. Vitamin A: supports immune function; found in orange/yellow vegetables. Iron: needed for red blood cell production after surgical blood loss; found in red meat, spinach, fortified cereals.

Hydration
Drink at least 6–8 glasses of water daily unless fluid-restricted. Adequate hydration is essential for kidney function (especially after anesthesia), wound healing, constipation prevention, and medication processing.

Managing Constipation Through Diet
Opioid medications and reduced activity commonly cause constipation. Preventive measures: increase fiber (whole grains, fruits, vegetables), drink plenty of water, consider stool softeners (docusate sodium) as directed, and walk as soon as allowed.

Foods to Avoid
In the early recovery period, avoid high-fat fried foods, spicy foods (if stomach is sensitive), carbonated beverages if bloating is present, alcohol (interacts with medications, impairs healing), and any foods your surgeon specifically instructed you to avoid.
""",

    "doc_06_medication_schedule.txt": """
MANAGING YOUR MEDICATIONS AFTER SURGERY

Overview
After surgery, you may be prescribed multiple medications with different schedules and instructions. Organizing and understanding your medications is critical to safe recovery.

Common Post-Surgery Medications
1. Pain medications (analgesics): opioids (oxycodone, hydrocodone), NSAIDs (ibuprofen, ketorolac), acetaminophen. 2. Antibiotics: prevent or treat surgical site infection; complete the full course. 3. Blood thinners (anticoagulants): prevent deep vein thrombosis (DVT) — aspirin, enoxaparin (Lovenox), or direct oral anticoagulants. 4. Anti-nausea medications (antiemetics): ondansetron, promethazine — take as needed. 5. Stool softeners/laxatives: prevent opioid-induced constipation; take docusate sodium with each opioid dose. 6. Proton pump inhibitors/antacids: protect stomach lining from NSAIDs.

Medication Safety Rules
Never skip doses without consulting your doctor. Always take medications with food unless specifically instructed otherwise. Never combine opioids with alcohol, benzodiazepines, or sleep aids without medical approval. Store medications in a secure, cool, dry place. Keep a complete medication list for all appointments and emergency situations.

Tapering Pain Medications
As pain improves (usually by week 2), switch from opioids to non-opioid pain relievers. Taper gradually rather than stopping abruptly to avoid withdrawal discomfort. Alert your doctor if you find it difficult to reduce opioid use.

Drug Interactions to Know About
Common interactions: NSAIDs + blood thinners → increased bleeding risk. Opioids + sedatives → increased respiratory depression. Antibiotics + blood thinners → altered levels of blood thinners. Always inform all healthcare providers and pharmacists of every medication you take.

Missed Doses
If you miss a scheduled dose of a pain medication, take it as soon as remembered unless it is nearly time for the next dose — never double-dose. For antibiotics, take missed doses as soon as remembered to maintain drug levels.

Medication Disposal
Unused opioids must be safely disposed of. Use pharmacy take-back programs, DEA-authorized collection sites, or medication disposal pouches. Never flush medications down the toilet unless specifically stated on the label.
""",

    "doc_07_follow_up_appointments.txt": """
FOLLOW-UP APPOINTMENTS AFTER SURGERY

Why Follow-Up Appointments Are Critical
Post-surgical follow-up appointments allow your surgeon or care team to: assess wound healing, remove sutures or staples, monitor for complications, adjust medications, and provide clearance for activity progression. Missing these appointments can result in undetected complications.

Typical Follow-Up Schedule
First appointment: 1–2 weeks post-surgery (wound check, suture/staple removal if applicable). Second appointment: 4–6 weeks post-surgery (functional assessment, activity clearance). Third appointment: 3 months post-surgery (long-term healing evaluation). Additional appointments may be scheduled based on your surgery type and individual progress.

What to Bring to Follow-Up Appointments
Your complete medication list (including over-the-counter drugs and supplements). A list of any symptoms, concerns, or questions you have. Records of your pain levels, temperature, and any unusual symptoms. Your discharge papers from the hospital or surgery center.

Questions to Ask at Your Follow-Up
When can I return to full activity? When can I return to work and/or driving? When can I shower, bathe, or swim? Are my incisions healing normally? Can I discontinue any medications? What symptoms should prompt me to call before my next appointment?

Telehealth Follow-Up Options
Many minor post-operative checks can now be conducted via telehealth video appointments. Take clear, well-lit photos of your incision for remote review. Telehealth is appropriate for medication management and symptom check-ins; in-person visits are required for physical examination, suture removal, and imaging.

When to Seek Care Between Appointments
Do not wait for a scheduled appointment if you experience: fever above 101.5°F, severe or worsening pain, wound opening or significant drainage, difficulty breathing, chest pain, swelling or redness in the legs, or any sudden new symptoms.
""",

    "doc_08_signs_of_infection.txt": """
RECOGNIZING AND RESPONDING TO SURGICAL SITE INFECTION

Overview of Surgical Site Infections (SSIs)
Surgical site infections occur when bacteria enter the surgical wound. SSIs affect 2–5% of all surgical patients and are a leading cause of post-operative complications. Early recognition and prompt treatment are essential to prevent SSIs from becoming serious.

Types of Surgical Site Infections
Superficial SSI: involves only the skin and subcutaneous tissue. Deep SSI: involves deeper tissue layers (fascia, muscle). Organ/space SSI: involves an organ or body cavity manipulated during surgery (most serious). Most SSIs occur within 30 days of surgery.

Early Warning Signs (Days 3–7)
Increasing redness spreading from the wound edges. Warmth that persists or worsens around the wound. Swelling that is increasing rather than decreasing. Drainage that becomes thick, colored (yellow, green, brown), or foul-smelling. Wound edges that begin to separate.

Systemic Signs of Infection
Fever above 101.5°F (38.6°C) is a key warning sign. Chills, body aches, and feeling generally unwell. Elevated heart rate. Confusion or disorientation (particularly in elderly patients, this may indicate serious infection or sepsis).

Normal vs. Abnormal Drainage
Normal: thin, clear or slightly blood-tinged fluid in the first 48 hours. Abnormal: purulent (thick, cloudy, colored) discharge; increasing volume; foul odor. Any purulent drainage from a surgical wound requires same-day medical evaluation.

Risk Factors for SSI
Diabetes mellitus (impairs immune function and wound healing). Obesity. Smoking. Immunosuppressive medications (corticosteroids, chemotherapy, biologic drugs). Malnutrition. Longer surgery duration. Emergency vs. elective surgery.

Treatment of SSIs
Treatment depends on severity. Superficial infections: wound opening (debridement), wound care, oral antibiotics. Deep infections: may require IV antibiotics, surgical re-exploration. Prevention is always preferred: proper preoperative preparation, sterile surgical technique, and careful post-operative wound care.

When to Go to the Emergency Room
Seek emergency care immediately if infection is accompanied by: rapidly spreading redness (especially in a streak pattern), high fever (above 103°F / 39.4°C), severe pain, hypotension (dizziness when standing), or signs of sepsis.
""",

    "doc_09_drain_care.txt": """
CARING FOR SURGICAL DRAINS AT HOME

What Are Surgical Drains?
Surgical drains are thin tubes inserted near a wound during surgery to remove excess fluid (blood, serum) that accumulates as the body heals. Drains prevent fluid collections (seromas or hematomas) that can harbor bacteria or delay healing. They are commonly used after breast surgery, abdominal surgery, orthopedic procedures, and plastic/reconstructive surgery.

Types of Drains
Jackson-Pratt (JP) drain: a flat, round bulb with a thin tubing. Hemovac drain: a larger, flat spring-loaded reservoir. Blake drain: ribbed tubing with no bulb. Your nurse will identify your drain type before discharge.

Daily Drain Care Routine
Empty the drain reservoir 2–3 times daily or when it reaches half full. Wash your hands thoroughly before and after drain care. Squeeze the drain bulb flat, close the plug, and allow the vacuum to reform. Record the volume and color of fluid drained each day — bring this log to your follow-up appointment.

Normal vs. Concerning Drain Output
Normal: Initially bloody, gradually turning pink, then yellow/clear over days. Volume typically decreases daily. Concerning: Sudden large increase in output; bright red blood (active bleeding); thick, purulent, or foul-smelling output; no output despite feeling of fullness around the drain site.

Preventing Drain Complications
Secure the drain tubing to your clothing with a safety pin or drain lanyard to prevent accidental pulling. Do not let the bulb drain into a toilet or sink without recording the volume first. Inspect the insertion site daily for redness, warmth, or drainage around the tube. Keep the insertion site covered with gauze and change the dressing daily.

When to Call Your Doctor
Call immediately if: drain output suddenly increases to over 100 mL/hour, output becomes bright red, the drain tube falls out, the insertion site becomes infected, or the drain stops producing fluid despite expected ongoing drainage.

Drain Removal
Your surgeon will remove the drain when daily output is consistently below 20–30 mL. This is usually performed in the office and takes only seconds — a brief stinging sensation is normal.
""",

    "doc_10_returning_to_work.txt": """
RETURNING TO WORK AFTER SURGERY

Factors That Determine Return-to-Work Timing
The appropriate time to return to work after surgery depends on several factors: the type and complexity of surgery performed, the physical demands of your job, your overall recovery progress, pain level and medication use (cannot safely drive or operate heavy equipment while taking opioids), and your employer's policies regarding medical leave.

Sedentary/Desk Work
For office work, computer work, or light administrative roles: most patients can return within 1–2 weeks for minor procedures (hernia repair, appendectomy, laparoscopic procedures). 2–4 weeks for major surgery (bowel resection, joint replacement, cardiac procedures). Remote work from home may be possible sooner if pain is controlled.

Moderate Physical Demands
For jobs involving moderate lifting (20–50 lbs), standing for long periods, or frequent movement: return is typically delayed to 4–6 weeks minimum. Clearance from your surgeon is required. Modified duty with lifting restrictions may be available as a transitional option.

Heavy Manual Labor
For construction, factory work, farming, or other physically demanding roles: full return typically requires 6–12 weeks or longer. Restrictions may include no lifting over 25 lbs, no bending/twisting, and no ladder climbing. Surgeons issue a "return-to-work release" letter documenting restrictions.

Driving Restrictions
Do not drive while taking opioid medications. Confirm with your surgeon when it is safe to resume driving — typically 1–2 weeks for minor surgery, longer for procedures affecting reaction time or the right foot (needed for braking).

Documentation for Employers
Request a disability or medical leave form from your surgeon. Keep your employer informed of expected return dates and any work restrictions. The Family and Medical Leave Act (FMLA) provides up to 12 weeks of unpaid, job-protected leave for eligible employees recovering from surgery.

Mental and Emotional Readiness
Physical readiness must be accompanied by mental readiness. Fatigue, pain, and medication side effects can impair focus and judgment. Return to work only when you can consistently perform your job duties safely.
""",

    "doc_11_knee_replacement_recovery.txt": """
RECOVERING FROM TOTAL KNEE REPLACEMENT SURGERY

Overview
Total knee replacement (TKR), also called total knee arthroplasty (TKA), involves replacing the damaged surfaces of the knee joint with metal and plastic components. It is one of the most commonly performed orthopedic surgeries in the United States, with over 700,000 procedures annually.

Hospital Stay
Most patients are hospitalized for 1–3 days. Physical therapy begins the day of or the day after surgery. You will be taught to walk with a walker and perform basic exercises before discharge.

First Two Weeks at Home
Keep the leg elevated when resting to minimize swelling. Apply ice packs for 15–20 minutes several times daily. Perform prescribed exercises — ankle pumps, quad sets, straight leg raises — to maintain circulation and prevent stiffness. Use a walker or crutches as directed; gradually transition to a cane.

Physical Therapy Schedule
Formal physical therapy typically begins within the first week and continues for 6–12 weeks. Sessions occur 2–3 times per week initially. Home exercise programs are prescribed for daily practice. Key exercises: range-of-motion (ROM) exercises, knee flexion/extension, hip strengthening, balance training.

Recovery Milestones
2 weeks: wound healing, reduced swelling, walking short distances. 6 weeks: transitioning off assistive devices, driving (if right knee — confirm with surgeon), returning to desk work. 3 months: most activities of daily living restored. 6–12 months: full recovery and final functional outcome.

Expected Pain and Swelling
Pain and swelling are most intense during the first 2–4 weeks. Persistent swelling can last 3–6 months — this is normal. Report significant increases in swelling, warmth, or pain to your surgeon.

Long-Term Outcomes
90% of total knee replacements last 15–20 years. Avoid high-impact activities (running, jumping) long-term. Preferred activities: walking, swimming, cycling, golf, low-impact aerobics.
""",

    "doc_12_hip_replacement_recovery.txt": """
RECOVERING FROM TOTAL HIP REPLACEMENT SURGERY

Overview
Total hip replacement (THR) involves replacing the femoral head (ball) and acetabulum (socket) with prosthetic components. It relieves pain and restores function in patients with severe hip arthritis, fracture, or osteonecrosis.

Hip Precautions — Critical Safety Rules
Following posterior approach hip replacement, you must follow hip precautions for 6–12 weeks to prevent dislocation: Do NOT bend the hip beyond 90 degrees. Do NOT cross your legs or ankles. Do NOT rotate the foot inward. Use elevated toilet seats and firm chairs to maintain safe hip angles.

First Week Recovery
Stay in bed with a pillow between your legs when lying on your side. Walk short distances with a walker under supervision. Perform prescribed ankle pumps and leg slides to prevent clots. Expect significant bruising, swelling, and moderate-severe pain.

Weeks 2–6
Progress from walker to cane as strength improves. Stairs: step up with the non-operated leg first, step down with the operated leg first ("up with the good, down with the bad"). Physical therapy focuses on hip strengthening, gait training, and balance.

Weeks 6–12
Hip precautions may be lifted (confirm with surgeon). Resume driving once cleared (typically 6–8 weeks for right hip replacement). Return to desk work typically at 4–6 weeks; physical labor at 3–6 months.

Signs of Hip Dislocation (Emergency)
Sudden severe hip pain after a fall or awkward movement. Leg rotation inward or appearing shorter than the other. Inability to bear weight. Visible deformity. Go to the emergency room immediately.

Long-Term Outcomes
Hip replacements typically last 20–25 years with modern implants. Avoid impact sports but stay active with low-impact exercise to maintain muscle strength around the implant.
""",

    "doc_13_appendectomy_recovery.txt": """
RECOVERY AFTER APPENDECTOMY

What is an Appendectomy?
An appendectomy is the surgical removal of the appendix, typically performed as an emergency procedure to treat appendicitis (acute inflammation of the appendix). Most appendectomies are performed laparoscopically (minimally invasive), with some requiring open surgery.

Laparoscopic vs. Open Appendectomy Recovery
Laparoscopic: 3 small incisions, 1–3 day hospital stay, return to normal activity in 2–4 weeks. Open: larger incision, 3–5 day hospital stay, return to normal activity in 4–6 weeks.

First Few Days at Home
Rest but get up to walk short distances every few hours. Eat light, easy-to-digest foods; advance diet as tolerated. Take prescribed pain medications as directed; transition to acetaminophen or ibuprofen as pain decreases. Keep incisions clean and dry.

Activity Restrictions
Avoid lifting over 10 lbs for 2–4 weeks. Avoid strenuous exercise, sports, or heavy work for 4–6 weeks. Light walking is encouraged from day 1. Driving: restricted while on narcotic pain medication; typically resume after 1 week.

Common Post-Appendectomy Symptoms
Shoulder or upper abdominal pain (referred pain from residual CO2 gas in laparoscopic surgery) — resolves in 24–48 hours. Fatigue is normal for 1–2 weeks. Mild constipation from opioid medications and reduced activity — use stool softeners.

Signs of Complications
Contact your surgeon if you experience: fever above 101.5°F, increasing abdominal pain, red or draining wound, inability to pass gas or stool for more than 3 days, nausea or vomiting preventing oral intake.

Return to Normal Life
Most patients return to desk work within 1 week. Full recovery and unrestricted activity typically occurs within 4–6 weeks for laparoscopic appendectomy.
""",

    "doc_14_hernia_repair_recovery.txt": """
RECOVERY AFTER HERNIA REPAIR SURGERY

Types of Hernia Repairs
Hernias occur when an organ or tissue protrudes through a weak spot in the surrounding muscle or connective tissue. Common types: inguinal (groin), umbilical, incisional (at previous incision site), hiatal (stomach/esophagus). Surgery may be open or laparoscopic; mesh is commonly used to reinforce the repair.

Hospital Stay and Immediate Post-Op
Most hernia repairs are outpatient (go home same day). Groin pain, swelling, and bruising are expected after inguinal hernia repair. Scrotal swelling is common in men after inguinal repair — use ice packs and supportive underwear.

Activity Restrictions
No lifting over 10–15 lbs for 4–6 weeks. Avoid straining during bowel movements (use stool softeners). No vigorous exercise or contact sports for 4–6 weeks. Driving: restricted while on opioid medications; typically 1–2 weeks for groin hernias.

Managing Pain
Significant groin or abdominal soreness is expected. An ice pack (20 minutes on, 20 off) reduces swelling significantly. Ibuprofen and acetaminophen are effective non-opioid options for mild-moderate pain.

Signs of Recurrence or Complications
New bulge at the repair site (possible hernia recurrence). Significant swelling, fever, or redness at the wound. Inability to urinate (urinary retention, a known complication of inguinal hernia repair). Numbness or tingling in the groin or upper thigh — usually temporary nerve irritation.

Long-Term Recovery
Most patients return to full activity by 6 weeks. Mesh repairs are generally durable. Avoid activities that chronically increase intra-abdominal pressure (heavy lifting, straining) to prevent recurrence.
""",

    "doc_15_gallbladder_removal_recovery.txt": """
RECOVERY AFTER LAPAROSCOPIC CHOLECYSTECTOMY (GALLBLADDER REMOVAL)

Overview
Laparoscopic cholecystectomy is the removal of the gallbladder through 3–4 small abdominal incisions. It is one of the most common surgeries performed in the US. Recovery is generally faster and less painful than open surgery.

Hospital Stay
Most patients are discharged the same day or the following morning. General anesthesia is used; you may feel groggy for 24 hours post-procedure.

First Week Recovery
Expect: right shoulder or upper back pain (referred pain from diaphragm irritation by CO2 gas — resolves within 48 hours), mild abdominal soreness, bloating, and fatigue. Nausea is common but usually short-lived. Diet: start with clear liquids and advance as tolerated. Some patients experience loose stools temporarily as the body adjusts to the absence of the gallbladder.

Dietary Adjustments
Without a gallbladder, bile drips continuously into the small intestine instead of being stored and released. Most patients tolerate a normal diet within 1–4 weeks. Some patients benefit from a low-fat diet for the first 2–4 weeks to reduce diarrhea. Gradually reintroduce higher-fat foods and note any individual intolerances.

Activity and Work Return
Light activity (walking) begins immediately. No lifting over 15 lbs for 2 weeks. Return to desk work typically in 1 week; physical labor in 2–4 weeks. Driving: safe once off opioid medications (typically 3–7 days).

Signs of Complications
Bile leak (rare but serious): increasing abdominal pain, fever, yellowing of skin/eyes (jaundice). Wound infection: increasing redness, swelling, drainage. Contact your surgeon promptly if these signs develop.
""",

    "doc_16_heart_bypass_recovery.txt": """
RECOVERING FROM CORONARY ARTERY BYPASS GRAFT (CABG) SURGERY

Overview
Coronary artery bypass graft (CABG) surgery redirects blood flow around blocked coronary arteries using blood vessels harvested from elsewhere in the body (usually the leg or chest wall). It is a major open-heart surgery requiring a median sternotomy (cutting through the breastbone).

Hospital Recovery (5–7 Days)
Time in the intensive care unit (ICU) for 1–2 days. Breathing tube removed typically within 12–24 hours of surgery. Progressive ambulation (walking) begins in the ICU. Multiple monitoring lines and chest drains are removed over days 1–3.

Sternal Precautions (Critical — First 6–8 Weeks)
The sternum (breastbone) is surgically split and held together with metal wires that heal over 6–8 weeks. During this healing period: Do NOT push or pull with arms to get up from a chair — use a heart pillow against the chest and roll to the side first. No lifting over 5–10 lbs. No driving. No strenuous arm activities. Sternal precautions prevent sternal wound complications (sternal dehiscence).

Managing Leg Incisions
If a saphenous vein from the leg was used as a graft, expect swelling, bruising, and soreness in the leg. Elevate the leg when sitting. Compression stockings reduce swelling. Walk regularly; this improves circulation and reduces leg swelling.

Cardiac Rehabilitation
A supervised cardiac rehabilitation program is strongly recommended starting 4–6 weeks after surgery. It includes: monitored exercise, risk factor education, stress management, and nutritional guidance. Participation significantly improves long-term cardiac outcomes.

Recovery Milestones
6 weeks: most daily activities resumed, driving permitted. 3 months: most patients return to work (sedentary jobs). 6–12 months: full recovery; return to exercise as guided by cardiac rehab.

Warning Signs After CABG
Chest pain, new or worsening shortness of breath (call 911). Sternal wound redness, drainage, or clicking sensation (call surgeon immediately — could indicate sternal instability). Leg swelling with calf pain (possible DVT). Fever above 101°F.
""",

    "doc_17_back_surgery_recovery.txt": """
RECOVERING FROM SPINAL SURGERY

Common Spinal Surgeries
Discectomy/microdiscectomy (removal of herniated disc material). Laminectomy (removal of the lamina to relieve spinal stenosis). Spinal fusion (joining two or more vertebrae). All require specific post-operative precautions and rehabilitation.

Immediate Post-Op (Hospital — 1–4 Days)
Pain and spasm are expected and managed with IV then oral medications. Physical therapists teach safe techniques for getting in and out of bed ("log roll" technique) and early walking. A brace or collar may be prescribed depending on the procedure.

Activity Restrictions — First 6 Weeks
Spine Precautions: avoid bending, lifting, and twisting (BLT) of the spine. Use proper body mechanics for all movements. Log roll technique for getting in and out of bed. Sit in firm, supportive chairs with arms. Lifting restricted to 5–10 lbs maximum.

Pain Management
Back pain improves gradually — do not expect immediate complete relief. Nerve pain (radiating leg pain, tingling, numbness) may take weeks to months to fully resolve as nerves recover. Ice for the first 72 hours; heat may be used after that for muscle spasm.

Physical Therapy
Formal PT typically begins at 4–6 weeks for fusion, sooner for discectomy/laminectomy. Focus on core strengthening, posture, and safe movement patterns. Aquatic therapy is often a gentle early option.

Return to Work and Driving
Desk work: typically 2–4 weeks for discectomy; 4–8 weeks for fusion. Heavy labor: 3–6 months or longer (fusion surgery). Driving: cleared by surgeon, usually 2–4 weeks after surgery.

Red Flags After Spinal Surgery
Cauda equina emergency: sudden bowel or bladder incontinence, numbness in the saddle area → call 911. Increasing leg weakness or worsening neurological symptoms. Fever with neck stiffness (meningitis — rare).
""",

    "doc_18_shoulder_surgery_recovery.txt": """
RECOVERING FROM SHOULDER SURGERY

Common Shoulder Procedures
Rotator cuff repair (suturing torn tendons), SLAP repair (superior labrum tear), shoulder labrum repair, total shoulder replacement, and shoulder arthroscopy for impingement. Recovery timelines vary significantly based on procedure complexity.

Immobilization Phase (Weeks 0–6)
Most shoulder surgeries require a sling for 4–6 weeks to protect the repair. Keep the arm in the sling at all times except for prescribed exercises and hygiene. Sleep with a pillow supporting the arm; many patients sleep in a recliner to keep the shoulder elevated. Do not lift anything with the surgical arm during this phase.

Physical Therapy Phases
Phase 1 (0–6 weeks): passive range of motion (therapist moves your arm) — pendulum exercises, passive forward flexion. Phase 2 (6–12 weeks): active-assisted and active range of motion. Phase 3 (12+ weeks): strengthening. Phase 4 (4–6 months): sport/occupation-specific training. Recovery is typically 4–6 months for rotator cuff repair; 1+ year for full return to sports.

Managing Pain and Swelling
Ice the shoulder for 15–20 minutes 3–4 times daily. Take prescribed pain medications as directed. Sleeping difficulty is extremely common after shoulder surgery — discuss with your surgeon if not improving.

Key Milestones
6 weeks: sling removal, begin active shoulder movement. 3 months: significant function restored, light activity. 6 months: return to most activities. 12 months: return to overhead sports (throwing, tennis, swimming).

When to Contact Your Surgeon
Sudden loss of motion after initial improvement (possible stiffness/frozen shoulder). Signs of infection. Significant mechanical symptoms (clicking, giving way, loud pops).
""",

    "doc_19_cataract_surgery_recovery.txt": """
RECOVERING FROM CATARACT SURGERY

Overview
Cataract surgery involves removing the cloudy lens of the eye and replacing it with a clear artificial intraocular lens (IOL). It is one of the most common and safest surgeries performed worldwide, typically done as an outpatient procedure taking 15–20 minutes.

Immediately After Surgery
You will go home with a protective eye shield. Vision may be blurry, hazy, or distorted for several hours to a few days. Colors may appear brighter than before surgery (the artificial lens is clearer). Mild eye discomfort, itching, or tearing is normal. Do NOT rub or press on the eye.

Eye Drop Regimen
You will be prescribed multiple eye drops (antibiotic, anti-inflammatory, steroid) to be used for 4–6 weeks. Follow the prescribed schedule precisely — usually several times per day at staggered intervals. Wash hands before applying drops. Wait at least 5 minutes between different drops.

Activity Restrictions
Avoid: bending over (head below waist level), heavy lifting (straining increases eye pressure), swimming or water contact with the eye for 2 weeks, dusty or smoky environments, strenuous activity for 1 week.

When Vision Should Improve
Most patients notice improved vision within 24–48 hours. Full visual stabilization occurs in 1–4 weeks as inflammation settles and the eye heals. New glasses may be prescribed 4–6 weeks post-surgery once vision has fully stabilized.

Signs of Complications
Seek emergency eye care for: sudden increase in pain (especially severe), sudden vision decrease, flashes of light or new floaters (may indicate retinal detachment), increasing redness not controlled by drops. Complications are rare but include infection (endophthalmitis), posterior capsule opacification (easily treated), and cystoid macular edema.
""",

    "doc_20_tonsillectomy_recovery.txt": """
RECOVERING FROM TONSILLECTOMY

What to Expect After Tonsillectomy
A tonsillectomy is the surgical removal of the tonsils, commonly performed to treat recurrent tonsillitis, sleep apnea, or tonsillar abscess. Recovery is known to be significantly more painful in adults than in children.

Pain — The Biggest Challenge
Pain peaks at days 5–7 after surgery (as scabs form and begin to fall off), then gradually improves. The pain is typically described as severe throat and ear pain. Ear pain (otalgia) is referred pain from the throat — this is normal. Take prescribed pain medications on a strict schedule, not just when pain is severe.

Scab Formation and Shedding
White or yellowish scabs form in the throat where the tonsils were — this is completely normal healing. Scabs shed between days 5–10 after surgery. Mild bleeding may occur at this time. DO NOT pick at or disturb the scabs.

Diet Guidelines
Hydration is the most important factor — dehydration is the leading cause of post-tonsillectomy readmission. Drink cool water, popsicles, cold apple juice, and soft foods. Cool and cold foods provide the most relief. Avoid: hot foods, spicy foods, crunchy or sharp-edged foods (crackers, chips, toast) until fully healed. Gradually advance diet as healing progresses.

Bleeding — A Serious Risk
Post-tonsillectomy bleeding (secondary hemorrhage) most commonly occurs on days 5–10. Warning signs: spitting bright red blood, blood clots, or more than a few teaspoons of blood. Go to the emergency room immediately if this occurs — do not wait.

Recovery Timeline
Days 1–4: moderate pain, white scab formation. Days 5–10: severe pain, scab shedding, highest risk of bleeding. Days 10–14: pain rapidly improves. Day 14–21: full healing.
""",

    "doc_21_colonoscopy_aftercare.txt": """
AFTERCARE FOLLOWING COLONOSCOPY

About Colonoscopy
A colonoscopy is a procedure in which a flexible scope with a camera is passed through the colon to examine the large intestine for polyps, cancer, inflammation, or other conditions. It typically takes 30–60 minutes.

Immediately After the Procedure
You will be in the recovery area for 30–60 minutes while sedation wears off. You MUST have a driver — you cannot drive yourself home after sedation. You may feel bloated, gassy, and have a mild sore throat (from any air insufflation).

Gas and Bloating
Bloating from the gas used to inflate the colon is common and resolves within hours. Walk around at home to help pass the gas more quickly. This is normal and not a sign of any problem.

When to Eat and Drink
Resume a normal diet as tolerated once fully awake and alert. Avoid alcohol for at least 24 hours. Stay well hydrated — you may be mildly dehydrated from the bowel prep.

If Polyps Were Removed (Polypectomy)
Mild cramping and small amounts of blood in the stool are normal for 1–2 days. Avoid NSAIDs (ibuprofen, aspirin, naproxen) for 7–10 days after polypectomy (increases bleeding risk). Avoid strenuous exercise for 24 hours. Diet: soft, easily digestible foods for 24 hours.

Warning Signs After Colonoscopy
Seek emergency care for: heavy rectal bleeding (more than a few tablespoons), severe abdominal pain or rigid abdomen (possible perforation), fever above 101°F, inability to pass gas or have a bowel movement within 24 hours.

Results and Next Steps
Biopsy results from polyps take 5–10 business days. Your physician will contact you with results and recommendations for your next colonoscopy.
""",

    "doc_22_c_section_recovery.txt": """
RECOVERING FROM CESAREAN SECTION (C-SECTION)

Overview
A cesarean section (C-section) is a surgical delivery of a baby through incisions in the abdomen and uterus. It is major abdominal surgery and requires more recovery time than vaginal delivery.

Hospital Recovery (3–4 Days)
Pain management: epidural or spinal anesthesia wears off gradually; IV pain medication transitions to oral medications. Early mobility: walking begins within 12–24 hours with nursing assistance. The urinary catheter is typically removed within 12–24 hours of surgery.

First Two Weeks at Home
Do not lift anything heavier than your newborn. Walk short distances several times daily to prevent blood clots. Rest whenever the baby sleeps. Avoid stairs if possible during the first week. Partner and family support is essential — you cannot do everything yourself.

Incision Care
The horizontal bikini-line incision is generally well-hidden and heals well. Keep the incision clean and dry; pat dry after showering. Avoid soaking (tub baths) for 3 weeks. Wearing high-waisted underwear can reduce rubbing on the incision.

Pain Management
Ibuprofen and acetaminophen are first-line options (safe while breastfeeding). Prescription opioids may be used for the first few days; taper quickly. Gas pain from surgery is common — walk and avoid carbonated beverages.

Activity Restrictions
No driving for 2–4 weeks (while on opioids; longer if reaction time impaired). No sex for 6 weeks. No strenuous exercise for 6–8 weeks.

Warning Signs
Heavy vaginal bleeding (soaking more than one pad per hour). Wound opening, significant discharge, increasing redness. Fever above 100.4°F. Difficulty breathing, chest pain, or calf pain (blood clot warning).

Emotional Health
Baby blues (mood swings, tearfulness) are common in the first 2 weeks. Postpartum depression is a serious condition affecting up to 15% of mothers — seek help if mood issues persist beyond 2 weeks.
""",

    "doc_23_thyroid_surgery_recovery.txt": """
RECOVERING FROM THYROID SURGERY (THYROIDECTOMY)

Overview
Thyroidectomy is the surgical removal of all or part of the thyroid gland. It is performed for thyroid cancer, goiter, hyperthyroidism, or suspicious nodules. A small horizontal incision is made at the base of the neck.

Immediate Post-Operative Period
Most patients are hospitalized for 1–2 days. Neck soreness and mild difficulty swallowing are expected. A small drain may be placed to collect fluid — usually removed before discharge.

Voice Changes
Temporary voice hoarseness is common due to irritation of the recurrent laryngeal nerve (runs adjacent to the thyroid). This usually resolves within days to weeks. Permanent voice change is a rare complication (less than 1%) of experienced thyroid surgeons.

Hypocalcemia — Low Calcium Levels
The parathyroid glands (which regulate calcium) are located near the thyroid and may be temporarily disrupted during surgery. Symptoms of low calcium: tingling around the mouth, fingertips, and toes; muscle cramps; anxiety; spasms. Calcium and vitamin D supplements are often prescribed preventively after total thyroidectomy.

Thyroid Hormone Replacement
After total thyroidectomy, lifelong levothyroxine (T4 replacement) is required. After partial thyroidectomy, hormone levels are monitored; replacement may or may not be needed. TSH levels are monitored every 6–12 weeks until stable, then annually.

Wound Care
The neck incision heals well when kept clean and dry. Silicone-based scar products can minimize the appearance of the scar once fully healed (after 3–4 weeks). Avoid sun exposure to the incision for 1 year to prevent hyperpigmentation.

Activity Restrictions
Avoid heavy lifting and strenuous neck movements for 2 weeks. Return to desk work: typically 1–2 weeks. Return to physical labor: typically 3–4 weeks. Driving: usually safe within 1 week if neck movement is not significantly restricted.
""",

    "doc_24_wrist_surgery_recovery.txt": """
RECOVERING FROM WRIST SURGERY

Common Wrist Surgeries
Carpal tunnel release (CTR): divides the transverse carpal ligament to relieve median nerve compression. Distal radius fracture fixation (ORIF): plates and screws stabilize a broken wrist. Wrist arthroscopy: minimally invasive diagnostic or therapeutic procedure. Triangular fibrocartilage complex (TFCC) repair.

Immobilization
Most wrist surgeries require a cast or splint for 2–8 weeks depending on the procedure. Carpal tunnel release: splint for 2–4 weeks. Fracture fixation (ORIF): cast/splint for 4–8 weeks. Keep the wrist elevated above heart level when resting to reduce swelling.

Hand Therapy (Occupational Therapy)
A hand/occupational therapist is your most important resource after wrist surgery. Therapy typically begins 1–4 weeks post-surgery and focuses on: swelling management, scar management, range-of-motion exercises, strengthening, and return-to-function tasks. Custom thermoplastic splints are commonly fabricated during therapy.

Pain and Swelling Management
Ice the wrist for 15–20 minutes 3–4 times daily (over a cloth — never direct contact). Keep the arm elevated. Take NSAIDs or prescribed pain medications as directed. Stiffness is common and improves with consistent exercises.

Return to Activity Timeline
Carpal tunnel release: return to light activities in 2–4 weeks; full function in 6–12 weeks. Distal radius ORIF: light activities at 6–8 weeks; full return to activity at 3–6 months.

Warning Signs
Significant color changes (white/blue/purple) in the fingers. Increasing numbness or tingling. Severe pain unresponsive to medications. Wound infection signs.
""",

    "doc_25_bariatric_surgery_recovery.txt": """
RECOVERING FROM BARIATRIC (WEIGHT LOSS) SURGERY

Overview of Bariatric Procedures
Common bariatric procedures include: Roux-en-Y gastric bypass (RYGB), sleeve gastrectomy, adjustable gastric band (lap band), and biliopancreatic diversion with duodenal switch (BPD/DS). All involve either restricting stomach size, limiting absorption, or both.

Hospital Stay
Typically 1–3 days for laparoscopic procedures. Early walking begins the day of or day after surgery.

Post-Operative Diet Progression
The diet after bariatric surgery is strictly staged: Stage 1 (Days 1–7): clear liquids only. Stage 2 (Weeks 2–4): full liquids — protein shakes, blended soups, milk. Stage 3 (Weeks 4–6): pureed foods — scrambled eggs, Greek yogurt, pureed meats. Stage 4 (Months 2–3): soft solid foods. Stage 5 (Month 3+): solid foods — advance slowly, avoid problem foods. Never skip stages; eating solid food too early can cause staple line leaks or severe vomiting.

Protein is Paramount
Minimum 60–80 grams of protein daily required. Protein preserves muscle mass during rapid weight loss. Protein supplements (shakes, powders) are typically necessary.

Vitamins and Supplements — Lifetime Requirement
All bariatric patients must take lifelong supplements: complete multivitamin with iron, calcium citrate (1200–1500 mg daily), vitamin D3 (3000 IU daily), vitamin B12. Bypass patients additionally need supplemental iron. Annual blood work monitors for nutritional deficiencies.

Common Post-Operative Issues
Dumping syndrome (RYGB): triggered by high-sugar or high-fat foods — sweating, nausea, rapid heart rate, diarrhea. Avoid by eating slowly, separating liquids from solids, and avoiding sugar. Dehydration: drink 64 oz of fluid daily; never drink during meals. Hair loss: common at 3–6 months post-op; resolves by 12–18 months.

Activity
Begin walking immediately. No strenuous exercise for 4–6 weeks. A lifelong exercise program is essential for long-term weight maintenance.
""",

    "doc_26_antibiotics_guide.txt": """
GUIDE TO ANTIBIOTICS AFTER SURGERY

Why Antibiotics Are Prescribed After Surgery
Antibiotics may be prescribed after surgery either as prophylaxis (to prevent infection in high-risk procedures) or as treatment (if an infection has been detected or is suspected). Post-surgical antibiotic use is carefully targeted to avoid contributing to antibiotic resistance.

Common Post-Surgical Antibiotics
Amoxicillin-clavulanate (Augmentin): broad-spectrum; used for soft tissue and wound infections. Cephalexin (Keflex): first-line for skin/wound infections; well-tolerated. Trimethoprim-sulfamethoxazole (TMP-SMX / Bactrim): used for MRSA-suspected infections. Metronidazole (Flagyl): covers anaerobic bacteria; used for abdominal surgeries. Ciprofloxacin/fluoroquinolones: used for urinary or abdominal infections.

Taking Antibiotics Correctly
Complete the full prescribed course, even if you feel better early. Stopping antibiotics early can allow bacteria to survive and develop resistance. Take at the same times each day to maintain stable blood levels. Take with food if gastrointestinal upset occurs.

Side Effects
Common: nausea, diarrhea, stomach upset. Take probiotics (Lactobacillus species) 2 hours after antibiotic doses to reduce GI side effects. C. difficile infection: a serious antibiotic-associated diarrheal illness — report severe, watery, or bloody diarrhea to your doctor. Allergic reactions: rash, itching, hives — seek medical attention. Anaphylaxis (rare): throat swelling, difficulty breathing — call 911.

Drug Interactions
Amoxicillin/penicillins: may reduce oral contraceptive effectiveness. Fluoroquinolones: may increase warfarin levels — close INR monitoring required. Metronidazole: severe reaction with alcohol — avoid all alcohol during treatment and for 48 hours after completing the course.

Antibiotic Resistance
Never share antibiotics or use leftover antibiotics from a prior prescription. Each antibiotic course should be specifically prescribed for your current infection.
""",

    "doc_27_opioid_pain_medication.txt": """
GUIDE TO OPIOID PAIN MEDICATIONS AFTER SURGERY

What Are Opioids?
Opioids are powerful prescription pain medications derived from or similar to natural opium compounds. They work by binding to opioid receptors in the brain and spinal cord to reduce pain perception. Common surgical opioids: oxycodone (OxyContin, Percocet), hydrocodone (Vicodin), morphine, tramadol, hydromorphone (Dilaudid), and codeine.

When and How to Take Opioids
Take only as prescribed — do not take extra doses. Take opioids for moderate-severe pain (5+/10); use acetaminophen or ibuprofen for mild pain. Taking opioids around the clock (scheduled) is more effective than waiting for severe pain. Reduce dose and frequency as pain improves; aim to transition to non-opioid analgesia within 3–7 days.

Common Side Effects
Constipation: take a stool softener (docusate sodium) with every opioid dose. Nausea: take with food; call doctor if severe. Drowsiness and impaired reaction time: do not drive or operate machinery. Itching: a common side effect; antihistamines can help. Dry mouth and urinary retention: stay hydrated; alert doctor if unable to urinate.

Opioid Safety Rules
Never combine with alcohol, benzodiazepines (Xanax, Valium), or sleeping pills — this combination can cause respiratory arrest. Store in a locked location away from children and others. Never share your prescription.

Recognizing Opioid Dependence
Physical dependence (adapts to the drug, experiences withdrawal if stopped) can develop within days. This is different from addiction. If you struggle to reduce or stop opioids, discuss with your doctor — tapering schedules and non-opioid options are available.

Safe Disposal
Never flush down the toilet (unless the label says to). Use DEA-authorized drug take-back programs. Pharmacy drop-boxes are widely available.
""",

    "doc_28_nsaids_ibuprofen_guide.txt": """
GUIDE TO NSAIDS (IBUPROFEN, NAPROXEN) AFTER SURGERY

What Are NSAIDs?
Non-steroidal anti-inflammatory drugs (NSAIDs) reduce pain by inhibiting cyclooxygenase (COX) enzymes, which produce prostaglandins — chemicals that cause inflammation, fever, and pain. Common NSAIDs: ibuprofen (Advil, Motrin), naproxen (Aleve, Naprosyn), ketorolac (Toradol — IV/IM for acute pain), celecoxib (Celebrex — COX-2 selective, less GI risk).

Benefits After Surgery
NSAIDs reduce: surgical inflammation, pain (comparable to opioids for mild-moderate pain), fever. Using NSAIDs reduces opioid requirements (opioid-sparing effect), decreasing opioid side effects and dependence risk.

When to Avoid NSAIDs
Your surgeon may restrict NSAIDs after: bone fracture surgery or spinal fusion (NSAIDs may impair bone healing), procedures with high bleeding risk, kidney problems (NSAIDs reduce renal blood flow), peptic ulcer disease, patients on blood thinners.

Taking NSAIDs Safely
Always take with food or milk to protect the stomach lining. Take with a proton pump inhibitor (omeprazole) if you have a history of stomach ulcers. Do not exceed the recommended dose or duration. Avoid combining multiple NSAIDs.

Side Effects
GI: stomach upset, nausea, ulcers (use lowest effective dose for shortest duration). Renal: fluid retention, reduced kidney function (particularly in elderly or dehydrated patients). Cardiovascular: slightly increased risk with long-term use; not a concern for short post-operative use. Bleeding: NSAIDs inhibit platelet function — avoid 7–10 days before and potentially after surgery (confirm with surgeon).

Ibuprofen vs. Acetaminophen
Both are effective non-opioid analgesics. They work by different mechanisms and can be safely combined for additive pain relief. Alternating ibuprofen and acetaminophen every 3 hours is an effective strategy for moderate pain.
""",

    "doc_29_blood_thinners_guide.txt": """
BLOOD THINNERS (ANTICOAGULANTS) AFTER SURGERY

Why Blood Thinners Are Prescribed
Surgery and immobility significantly increase the risk of deep vein thrombosis (DVT — blood clots in leg veins) and pulmonary embolism (PE — clots in the lungs). Blood thinners reduce this risk during the high-risk post-surgical period.

Types of Blood Thinners
Heparin (subcutaneous injections): used in hospital, fast-acting, easily reversible. Low molecular weight heparin (LMWH): enoxaparin (Lovenox), dalteparin — daily or twice-daily self-injections; very effective DVT prophylaxis. Warfarin (Coumadin): oral; requires regular INR blood monitoring. Direct oral anticoagulants (DOACs): rivaroxaban (Xarelto), apixaban (Eliquis), dabigatran (Pradaxa) — oral, no routine monitoring required; increasingly preferred.

Self-Injecting Enoxaparin at Home
Use the prefilled syringe. Choose the abdomen (2 inches from navel) or outer thigh. Clean skin with alcohol, pinch fat, insert needle at 90 degrees, inject slowly, do not rub after withdrawal. Rotate injection sites to avoid bruising. Dispose of sharps in a sharps container.

Monitoring and Precautions
Report unusual bruising, bleeding from gums, blood in urine or stool, heavy menstrual bleeding, or prolonged bleeding from cuts. Avoid contact sports and activities with fall risk. Take extra care when shaving.

Duration of Therapy
After joint replacement surgery: typically 10–35 days. After abdominal surgery: typically 7–14 days. Higher-risk patients may require longer courses.

Drug and Food Interactions (Warfarin)
Vitamin K (leafy green vegetables) reduces warfarin effectiveness. Consistent vitamin K intake is important — do not drastically change diet. Many antibiotics, antifungals, and herbal supplements interact with warfarin — always inform your prescriber.
""",

    "doc_30_anti_nausea_medications.txt": """
MANAGING NAUSEA AND VOMITING AFTER SURGERY

Why Nausea Occurs After Surgery
Post-operative nausea and vomiting (PONV) affects 20–30% of surgical patients. Causes include: general anesthesia effects (especially volatile anesthetics), opioid pain medications, bowel manipulation during abdominal surgery, fasting state, and individual susceptibility (female sex, history of motion sickness, non-smoker status all increase PONV risk).

Medications for PONV
Ondansetron (Zofran): first-line antiemetic; serotonin receptor antagonist; no sedation. Promethazine (Phenergan): antihistamine; effective but causes significant sedation. Metoclopramide (Reglan): prokinetic; helps the stomach empty faster; useful for nausea with delayed gastric emptying. Scopolamine patch (Transderm Scop): applied behind the ear before surgery; effective for 72 hours. Dexamethasone: steroid given during surgery; reduces PONV significantly.

Non-Medication Strategies
Acupressure bands on the wrist (P6 point): scientifically supported; non-invasive. Cold, bland foods (crackers, toast, applesauce) are better tolerated when nauseous. Avoid strong odors — cooking smells, perfumes. Rest in a semi-reclined position rather than lying completely flat. Take pain medications with food.

When to Call Your Doctor
Severe vomiting preventing oral medication intake. Vomiting for more than 24 hours. Signs of dehydration: dry mouth, no urination for 8+ hours, severe dizziness. Blood in vomit (hematemesis).

Preventing Dehydration
If you cannot keep solid food down, focus on: small frequent sips of clear liquids (water, broth, flat ginger ale, popsicles). Oral rehydration solutions (Pedialyte) if vomiting is significant. Avoid full glasses at once — small sips every 5–10 minutes.
""",

    "doc_31_stool_softeners_laxatives.txt": """
MANAGING CONSTIPATION AFTER SURGERY

Why Constipation Occurs After Surgery
Post-surgical constipation is extremely common and has multiple contributing causes: opioid pain medications (reduce bowel motility), anesthesia effects (slow the gut temporarily), reduced physical activity, dehydration from fasting and fluid losses, dietary changes (reduced fiber, liquid diet), and psychological stress.

Constipation Prevention — Start Immediately
Begin stool softeners with the first opioid dose — do not wait for constipation to develop. Drink plenty of fluids (8+ cups daily). Walk as much as allowed. Eat fiber-containing foods as soon as your diet allows (fruits, vegetables, whole grains).

Stool Softeners
Docusate sodium (Colace): softens stool by drawing water into it; no stimulant effect; very safe; start with 100 mg twice daily. Polyethylene glycol (MiraLax): tasteless powder mixed with water; osmotic laxative; highly effective; gentle. These are first-line options and safe for daily use during opioid therapy.

Stimulant Laxatives
Bisacodyl (Dulcolax), senna: stimulate bowel contractions; use if stool softeners alone are insufficient; may cause cramping. Milk of magnesia (magnesium hydroxide): osmotic; effective for acute relief.

When to Escalate
If no bowel movement for 3 days despite stool softeners, take a stimulant laxative. If no bowel movement for 5 days, contact your surgeon. Go to the emergency room if: severe abdominal distension, inability to pass gas, nausea and vomiting with abdominal pain (possible ileus or bowel obstruction).

Non-Medication Approaches
Warm liquids (coffee, prune juice, warm water with lemon) stimulate bowel activity. Abdominal massage in a clockwise direction. Squatting position (using a step stool under feet on the toilet) aids elimination.
""",

    "doc_32_antacids_ppis.txt": """
STOMACH PROTECTION: ANTACIDS AND PROTON PUMP INHIBITORS

Why Stomach Protection Is Needed After Surgery
NSAIDs (ibuprofen, naproxen, ketorolac) commonly used for post-operative pain inhibit prostaglandins that protect the stomach lining, increasing the risk of gastric ulcers and GI bleeding. Proton pump inhibitors (PPIs) or antacids counteract this risk.

Proton Pump Inhibitors (PPIs)
Mechanism: block the proton pump in the stomach, dramatically reducing acid production. Common PPIs: omeprazole (Prilosec), pantoprazole (Protonix), esomeprazole (Nexium), lansoprazole (Prevacid). Use: take 30–60 minutes before the first meal of the day. When to use: prescribed with NSAIDs, especially for patients with history of ulcers, elderly patients, or those on blood thinners.

H2 Blockers
Mechanism: block histamine H2 receptors, reducing acid production (less potent than PPIs). Examples: famotidine (Pepcid), ranitidine (recalled due to contaminant concerns). Use: taken twice daily or as needed for heartburn.

Antacids
Mechanism: neutralize existing stomach acid. Examples: Tums (calcium carbonate), Maalox, Mylanta. Use: for immediate relief of heartburn or GI upset. Important: antacids can interfere with absorption of some medications (antibiotics, thyroid medications) — take 2 hours apart.

Duration of Therapy
For patients taking NSAIDs: continue PPI for the duration of NSAID use. For patients with significant stress ulcer risk (ICU, mechanical ventilation): ongoing PPI during hospitalization. Not intended for indefinite use — discuss stopping with your doctor.

Side Effects of PPIs (Long-Term Use)
Reduced calcium and magnesium absorption (relevance if used for months-years). Increased risk of Clostridioides difficile infection with prolonged use. For short post-operative courses (days-weeks), PPIs are very safe.
""",

    "doc_33_sleep_aids_after_surgery.txt": """
MANAGING SLEEP AFTER SURGERY

Why Sleep is Disrupted After Surgery
Post-operative sleep disruption is extremely common due to: pain and discomfort, hospital environment (noise, vital signs checks), anxiety, opioid medications (disrupt REM sleep), physical immobility, and circadian rhythm disturbance.

Why Good Sleep Is Critical for Recovery
Sleep is when the body produces the most growth hormone, initiates tissue repair, consolidates immune function, and processes anesthesia metabolites. Sleep deprivation impairs wound healing, immune response, pain tolerance, and emotional well-being.

Non-Pharmacological Sleep Strategies
Sleep hygiene: same bedtime and wake time daily. Reduce light exposure at night (blackout curtains). Comfortable positioning: use pillows to support surgical site and reduce pain. Relaxation techniques: deep breathing, progressive muscle relaxation, guided imagery. Limit caffeine after noon.

Medications for Post-Surgical Sleep
Melatonin (0.5–5 mg at bedtime): regulates sleep-wake cycle; non-habit-forming; minimal side effects; first choice. Diphenhydramine (Benadryl, 25–50 mg): OTC sleep aid; causes next-day grogginess; may cause urinary retention in men. Trazodone (25–100 mg): prescription; sedating antidepressant frequently used for sleep; not habit-forming. Benzodiazepines/zolpidem: prescription; effective short-term, but increase fall risk in post-surgical patients and interact with opioids — generally avoided unless absolutely necessary.

Avoid Alcohol for Sleep
Alcohol may help with initial sleep onset but disrupts REM sleep architecture and causes early morning awakening.

When to Address Persistent Sleep Problems
If sleep remains poor after 2–3 weeks despite good sleep hygiene, discuss with your doctor. Cognitive behavioral therapy for insomnia (CBT-I) is the gold-standard long-term treatment.
""",

    "doc_34_topical_treatments.txt": """
TOPICAL TREATMENTS FOR WOUND CARE AND SCAR MANAGEMENT

Topical Antibiotics
Bacitracin ointment: applied to fresh incisions to prevent superficial infection; reduces crust formation; use thin layer once daily. Mupirocin (Bactroban): prescription; specific for Staphylococcus and Streptococcus; used for infected wounds or to decolonize MRSA carriers. Important: topical antibiotics should not be used indefinitely — typically 5–7 days unless directed by your surgeon.

Wound Healing Agents
Petrolatum (Vaseline): simple, effective, and inexpensive; keeps wounds moist, reducing scar formation; preferred over triple antibiotic ointment by many surgeons. Hydrogel dressings: donate moisture to dry wounds. Foam dressings: absorb excess drainage from wounds with heavy exudate.

Scar Management Topicals (After Wound Fully Closed)
Silicone gel sheeting or silicone gel: gold standard for scar prevention and treatment. Apply to fully closed wounds for 12 hours/day minimum for 3–6 months. Reduces scar elevation, firmness, and discoloration. Onion extract gels (Mederma): some evidence for scar appearance improvement; less effective than silicone. Vitamin E oil: commonly used but evidence is mixed; may cause contact dermatitis in some patients.

Sunscreen for Scars
Healing scars are hypersensitive to UV radiation, which can cause permanent hyperpigmentation. Apply SPF 30+ broad-spectrum sunscreen to scars whenever exposed to sunlight. Continue sun protection for 12–18 months after surgery.

Topical Anesthetics
Lidocaine gel (2–4%): applied to wounds before dressing changes to reduce procedural pain. Available over-the-counter as a topical anesthetic.
""",

    "doc_35_vitamins_supplements_recovery.txt": """
VITAMINS AND SUPPLEMENTS TO SUPPORT SURGICAL RECOVERY

The Role of Nutrition in Healing
Optimal nutrition accelerates every phase of wound healing: inflammation resolution, collagen synthesis, tissue remodeling, and immune defense. Targeted supplementation can fill dietary gaps that are common post-surgery due to reduced appetite and dietary restrictions.

Key Supplements for Recovery

Vitamin C (500–1000 mg/day)
Vitamin C is essential for collagen synthesis — the protein that forms the structural framework of healing tissue. It is also a powerful antioxidant that reduces oxidative stress from surgical trauma. Deficiency impairs wound healing significantly. Food sources: citrus fruits, bell peppers, kiwi, strawberries.

Zinc (25–45 mg/day for short-term use)
Zinc accelerates wound healing by supporting cell division, protein synthesis, and immune function. Zinc deficiency is common in surgical patients due to increased zinc losses. Do not exceed 40 mg daily long-term (interferes with copper absorption).

Vitamin D (1000–4000 IU/day)
Critical for immune function and bone healing (especially relevant after orthopedic surgery). Deficiency (very common) impairs fracture healing and increases infection susceptibility.

Protein Supplements
If unable to meet protein needs (1.2–1.5 g/kg/day) through diet alone, whey protein or pea protein supplements can fill the gap.

Omega-3 Fatty Acids (Fish Oil)
Anti-inflammatory effects support recovery, but should be STOPPED 2 weeks before surgery (mild antiplatelet effect). Can be restarted 2 weeks after surgery if indicated.

What to Avoid
Avoid high-dose antioxidants (Vitamins E >400 IU, C >2000 mg) in the first week after surgery — paradoxically, excessive antioxidant supplementation may impair the necessary inflammatory phase of healing.

Herbal Supplements to Avoid After Surgery
Garlic, ginger, ginseng, St. John's Wort, kava: may interfere with anesthesia, clotting, or prescribed medications. Discuss all supplements with your surgeon.
""",

    "doc_36_deep_vein_thrombosis.txt": """
DEEP VEIN THROMBOSIS (DVT) — RECOGNITION AND PREVENTION

What Is DVT?
Deep vein thrombosis is a blood clot that forms within a deep vein, typically in the calf, thigh, or pelvis. DVT is a serious post-surgical complication because clots can travel to the lungs (pulmonary embolism), potentially causing sudden death.

Why Surgery Increases DVT Risk
Virchow's triad describes the three factors that contribute to DVT: venous stasis (reduced blood flow from immobility), endothelial injury (from surgery), and hypercoagulability (the body's pro-clotting response to surgery). The combination after surgery is a perfect storm for clot formation.

High-Risk Surgeries
Total hip replacement (highest risk). Total knee replacement. Major abdominal/pelvic surgery. Cancer surgery. Prolonged surgery (> 2 hours). Any surgery in patients with prior DVT, obesity, cancer, or clotting disorders.

Symptoms of DVT
Leg swelling (typically one leg). Calf pain or tenderness — may feel like a pulled muscle. Warmth and redness in the affected leg. In some cases, DVT causes no symptoms ("silent DVT").

Prevention Strategies
Blood thinners (anticoagulants): enoxaparin, rivaroxaban, apixaban — most important prevention strategy. Sequential compression devices (SCDs): inflatable leg sleeves worn in hospital. Compression stockings: grade II stockings reduce venous stasis. Early mobilization: walking within hours of surgery — most effective non-pharmacological measure. Hydration: stay well-hydrated; dehydration thickens blood.

Treatment of DVT
Anticoagulation for 3–6 months. Compression stockings to reduce post-thrombotic syndrome. Rarely, thrombolytic therapy or mechanical thrombectomy for massive DVT with limb threat.

When to Seek Emergency Care
Sudden chest pain or shortness of breath (possible pulmonary embolism) → Call 911 immediately. One-sided leg swelling with pain → seek same-day evaluation.
""",

    "doc_37_pulmonary_embolism.txt": """
PULMONARY EMBOLISM (PE) — RECOGNIZING A LIFE-THREATENING EMERGENCY

What Is Pulmonary Embolism?
A pulmonary embolism occurs when a blood clot (usually from a deep vein thrombosis) travels through the bloodstream and lodges in the pulmonary arteries, blocking blood flow to part of the lung. PE is one of the leading preventable causes of hospital death.

Risk After Surgery
PE risk is highest in the first 4–6 weeks after major surgery, with peak risk in the first 2 weeks. The risk persists beyond discharge — most fatal PEs after joint replacement occur at home.

Symptoms of Pulmonary Embolism
Sudden shortness of breath (most common symptom). Sharp chest pain that worsens with breathing (pleuritic chest pain). Rapid heart rate. Low oxygen saturation. Coughing up blood (hemoptysis). Anxiety, sense of impending doom. Fainting or near-fainting.

CALL 911 IMMEDIATELY
Pulmonary embolism is a medical emergency. Do not drive yourself to the hospital. Call 911. Every minute of delay worsens outcomes.

Diagnosis
In the emergency department: CT pulmonary angiography (CTPA) — gold standard diagnostic test. D-dimer blood test: high sensitivity for PE; negative result effectively rules out PE. V/Q scan: nuclear medicine scan used when CT is contraindicated. Echocardiogram: assesses right heart strain from PE.

Treatment
Anticoagulation: LMWH or direct oral anticoagulants for 3–6 months minimum. Thrombolytics (clot-busting drugs): reserved for massive PE with cardiovascular instability. Catheter-directed therapy or surgical embolectomy: for life-threatening massive PE.

Prevention
Prevention of PE is identical to DVT prevention (see DVT document) — the two conditions are intimately linked.
""",

    "doc_38_surgical_site_infection.txt": """
SURGICAL SITE INFECTION (SSI) — PREVENTION AND MANAGEMENT

Classification of SSIs
According to the CDC, SSIs are classified as:
Superficial incisional SSI: involves only skin and subcutaneous tissue. Deep incisional SSI: involves deep soft tissues (fascia, muscle). Organ/space SSI: affects organs or cavities opened during surgery (e.g., peritonitis after abdominal surgery).

Causative Organisms
Most SSIs are caused by Staphylococcus aureus (including MRSA), Streptococcus species, and gram-negative rods (E. coli, Klebsiella, Pseudomonas). Polymicrobial infections are common after bowel surgery.

Timing of SSI Development
Most SSIs present 5–10 days after surgery, though deep or organ SSIs can present later. SSIs occurring within 30 days of surgery (or within 1 year if an implant was placed) are classified as surgical site infections.

Diagnosis
Clinical signs: redness, warmth, swelling, purulent drainage, wound opening, fever. Laboratory: wound culture, CBC (elevated white blood cell count), inflammatory markers (CRP, ESR). Imaging: ultrasound or CT scan to detect deep fluid collections (abscess).

Treatment
Superficial SSI: wound opening and irrigation (by surgeon), wound care, oral antibiotics. Deep SSI: IV antibiotics, surgical debridement and drainage. Abscess: incision and drainage (I&D). Implant infection: often requires implant removal — devastating complication after joint replacement.

Prevention
Preoperative: skin preparation with chlorhexidine scrub the night before surgery, preoperative antibiotics within 60 minutes of incision. Intraoperative: sterile technique, maintaining normal body temperature and blood glucose. Postoperative: proper wound care, early detection and treatment of SSI.
""",

    "doc_39_internal_bleeding_signs.txt": """
RECOGNIZING POST-OPERATIVE BLEEDING COMPLICATIONS

Overview
Bleeding after surgery can be internal (hidden) or external (visible at the wound). All post-operative patients and caregivers should know the warning signs, as timely recognition can be life-saving.

Normal Post-Operative Bleeding
A small amount of bleeding or oozing at the incision site in the first 24–48 hours is expected. Bruising (ecchymosis) around the incision and in dependent areas (from blood tracking through tissue) is normal and may spread before clearing.

Signs of Excessive External Bleeding
Soaking through a dressing within 1 hour requires medical evaluation. Bright red, pulsatile blood from the wound. Drain output suddenly turning bright red or exceeding 100 mL/hour.

Signs of Internal Bleeding
Internal bleeding is more dangerous because it is not visible until serious compromise occurs. Warning signs include: rapid heart rate (tachycardia), significant drop in blood pressure (hypotension), dizziness, lightheadedness, or fainting on standing. Pallor (pale skin, pale lips). Abdominal pain and distension after abdominal surgery. Swelling of a limb out of proportion to expected surgical swelling. Shortness of breath or chest pain.

Medications That Increase Bleeding Risk
Blood thinners (anticoagulants, antiplatelet agents): all increase bleeding risk. NSAIDs: inhibit platelet function for 7–10 days. Fish oil, vitamin E, garlic supplements: mild antiplatelet effects.

When to Seek Emergency Care
Call 911 or go to the emergency room if you experience: heavy wound bleeding, dizziness and rapid heart rate, sudden severe abdominal pain, shortness of breath with chest pain, or collapse.

At the Emergency Room
IV access will be established. Blood tests (CBC, coagulation studies, type and cross-match for possible transfusion). Imaging (CT scan, ultrasound) to identify bleeding source. Intervention: endoscopy, angioembolization, or return to the operating room.
""",

    "doc_40_pneumonia_after_surgery.txt": """
PNEUMONIA AND ATELECTASIS AFTER SURGERY

Why Lung Complications Occur After Surgery
Post-operative pulmonary complications are among the most common serious complications, affecting 2–19% of surgical patients. Pain from thoracic or abdominal incisions causes patients to take shallow breaths, avoiding deep inspiration. Shallow breathing leads to atelectasis (collapse of small lung segments), which can progress to pneumonia.

Atelectasis
Atelectasis is the partial or complete collapse of lung tissue. It occurs because patients breathe shallowly post-operatively, allowing mucus to accumulate and small airways to close. Mild atelectasis typically presents as a low-grade fever (100–101°F) in the first 24–48 hours and often resolves with increased activity and deep breathing.

Preventing Pneumonia — Deep Breathing Exercises
Incentive spirometry: a handheld device that encourages deep breaths. Use 10 times every hour while awake. Insert mouthpiece, exhale fully, then inhale slowly and deeply, raising the ball/column. This is the single most important intervention for preventing post-op pneumonia.

Coughing Exercises
Productive coughing clears secretions from the airways. Use the splinting technique (pillow against chest or abdomen) to cough effectively without incision pain. Controlled coughing: two short coughs followed by one forceful cough.

Early Ambulation
Walking is one of the best ways to prevent lung complications after surgery. The physical activity and upright position promote lung expansion, improve circulation, and stimulate mucus clearance. Start walking as soon as your surgical team permits.

Symptoms of Pneumonia
New or worsening cough with thick discolored sputum. Fever above 101.5°F occurring after day 2 (early fever is often atelectasis; later fever suggests infection). Shortness of breath. Chest pain with breathing (pleuritic pain). Confusion (especially in elderly patients).

Treatment
Mild pneumonia: oral antibiotics, increased ambulation, respiratory therapy. Severe pneumonia: hospital admission, IV antibiotics, supplemental oxygen.
""",

    "doc_41_urinary_tract_infection.txt": """
URINARY TRACT INFECTIONS (UTIs) AFTER SURGERY

Why UTIs Are Common After Surgery
Urinary tract infections are one of the most common post-surgical complications. Primary cause: urinary catheters (Foley catheters) placed during surgery allow bacteria direct access to the bladder. Risk increases with catheter duration.

Symptoms of UTI
Urinary frequency and urgency. Burning or pain during urination (dysuria). Cloudy, dark, or foul-smelling urine. Lower abdominal or pelvic discomfort. Fever (usually below 101.5°F for lower UTI; higher for upper UTI/pyelonephritis). In elderly patients: UTI may present as confusion or altered mental status without classical urinary symptoms.

Diagnosis
Urine analysis (UA): detects white blood cells (pyuria) and bacteria (bacteriuria). Urine culture: identifies the specific bacteria and antibiotic sensitivities. Dipstick UA (positive leukocyte esterase/nitrites): rapid screening tool.

Treatment
Lower UTI (bladder infection/cystitis): 3–7 day course of antibiotics (trimethoprim-sulfamethoxazole, nitrofurantoin, fosfomycin). Upper UTI (kidney infection/pyelonephritis): 10–14 day antibiotic course; may require hospitalization.

Prevention Strategies
Catheter removal as early as surgically appropriate (ideally within 24–48 hours of surgery). Adequate hydration (promotes frequent urination, flushing bacteria). Proper hygiene. In women: wipe front to back after toileting. Avoid prolonged catheterization.

When to Call Your Doctor
Symptoms of UTI require evaluation and treatment — do not ignore them. Untreated UTI can progress to kidney infection (pyelonephritis) or, in rare cases, sepsis.
""",

    "doc_42_anesthesia_side_effects.txt": """
ANESTHESIA SIDE EFFECTS AND RECOVERY

Types of Anesthesia
General anesthesia: renders the patient completely unconscious; used for major surgery. Regional anesthesia: numbs a large area (e.g., spinal, epidural, nerve blocks); patient may remain conscious. Local anesthesia: numbs a small area; used for minor procedures. Monitored anesthesia care (MAC)/sedation: IV sedation with local anesthesia; used for procedures like colonoscopy.

Common Side Effects of General Anesthesia
Nausea and vomiting (PONV): affects 20–30% of patients; preventive medications often given intraoperatively. Sore throat: from endotracheal intubation; usually resolves in 1–2 days; ice chips and throat lozenges help. Drowsiness and confusion: normal for 24–48 hours, especially in elderly patients (POCD — post-operative cognitive dysfunction). Shivering: as body temperature normalizes after surgery; warm blankets are provided. Dry mouth.

Regional Anesthesia Side Effects
Spinal/epidural: temporary leg weakness, urinary retention, headache (post-dural puncture headache — positional, worsened by sitting up), backache. Nerve blocks: temporary numbness and weakness in the blocked region; protect the numb limb from injury. Nerve damage: rare (<1%) but can cause prolonged numbness or weakness.

Serious (Rare) Anesthesia Complications
Malignant hyperthermia: life-threatening reaction to certain anesthetic gases; treated with dantrolene. Anaphylaxis: severe allergic reaction to anesthetic agents. Aspiration pneumonia: inhalation of stomach contents; prevented by fasting before surgery. Post-operative cognitive dysfunction (POCD): prolonged cognitive impairment, mainly in elderly patients; usually transient.

Resuming Medications After Anesthesia
Most regular medications can be resumed with small sips of water once you are fully awake and can swallow safely. Medications withheld before surgery (blood thinners, diabetes medications) — timing of restart should be confirmed with your surgical team.
""",

    "doc_43_breathing_exercises.txt": """
BREATHING EXERCISES AFTER SURGERY

Why Breathing Exercises Are Critical
Following surgery — especially abdominal, thoracic, or any surgery requiring general anesthesia — lung function is temporarily impaired. Breathing exercises prevent atelectasis (lung collapse), pneumonia, and post-operative fever. They are among the most important things you can do for your recovery.

Incentive Spirometry
An incentive spirometer is a small device that provides feedback on the depth of your breaths. Instructions: sit upright in a chair or at the head of the bed. Exhale normally. Place the mouthpiece between your lips and create a seal. Inhale slowly and deeply to raise the piston as high as possible. Hold for 3–5 seconds. Exhale. Goal: perform 10 times every hour while awake. Clean the mouthpiece with soap and water daily.

Diaphragmatic Breathing
Also called abdominal breathing. Procedure: lie comfortably or sit upright. Place one hand on your chest, one on your abdomen. Breathe in through the nose, letting your abdomen expand (lower hand rises). Breathe out through pursed lips, letting the abdomen fall. The hand on the chest should remain relatively still. Benefits: maximizes air entry to the lower lung lobes, reduces anxiety, and improves oxygen exchange.

Pursed-Lip Breathing
Inhale slowly through the nose for 2 counts. Pucker lips as if blowing out a candle. Exhale slowly and completely for 4 counts. Prevents airway collapse during exhalation; slows breathing rate; reduces work of breathing.

Coughing Exercises
"Huff cough" technique: take a deep breath through the mouth, then forcefully exhale saying "huff, huff." Therapeutic coughing: deep breath, then 2–3 sharp coughs. Use a pillow to splint painful abdominal or chest incisions.

Frequency and Progression
Initially: breathing exercises every hour while awake. As recovery progresses: perform 10–15 deep breaths every time you change position (sitting up from bed, standing up from chair).
""",

    "doc_44_walking_after_surgery.txt": """
WALKING AND EARLY MOBILIZATION AFTER SURGERY

Why Early Walking Is So Important
Early ambulation (walking) after surgery is one of the most powerful interventions in post-operative recovery. Benefits include: prevents deep vein thrombosis (blood clots), prevents pneumonia and atelectasis, maintains muscle strength and prevents muscle wasting, improves bowel function and reduces constipation, reduces hospital length of stay, accelerates overall recovery.

When to Start Walking
Most patients are encouraged to stand or walk within 4–24 hours of surgery, depending on: procedure type, anesthesia type, vital sign stability, and pain level. Your nurse or physical therapist will guide your first attempts.

First Steps — Safety Guidelines
Always have assistance for the first few times. Sit on the edge of the bed for 1–2 minutes before standing (prevents orthostatic hypotension/dizziness). Rise slowly. If dizzy, sit back down and wait. Use an assistive device (walker, cane) if recommended. Wear non-slip footwear — never walk in socks alone on a hospital floor.

Avoiding Orthostatic Hypotension
Blood pressure drops when changing from lying to standing after surgery are common due to: bed rest, fluid shifts, pain medications, and anesthesia effects. Management: rise slowly in stages (lying → sitting → standing), stay hydrated, tell staff immediately if you feel faint.

Weekly Walking Progression (Post-Discharge)
Week 1: 2–3 walks per day, 5–10 minutes each, within the house. Week 2: increase to 15–20 minutes per walk. Week 3–4: increase to 20–30 minutes. Longer per surgical team clearance and individual tolerance.

Signs to Stop and Rest
Stop walking immediately if you experience: chest pain, severe shortness of breath, dizziness or near-fainting, sudden severe pain at the surgical site.
""",

    "doc_45_physical_therapy_basics.txt": """
PHYSICAL THERAPY AFTER SURGERY — WHAT TO EXPECT

Role of Physical Therapy
Physical therapy (PT) is a critical component of recovery from many types of surgery, especially orthopedic (joint replacement, fracture repair, spinal surgery), cardiac (bypass surgery), abdominal, and thoracic procedures. Physical therapists design individualized exercise programs to restore strength, flexibility, balance, and function.

When PT Begins
In-hospital PT: often begins the day of or day after surgery. Outpatient PT: typically starts 1–2 weeks after discharge. Home PT: for patients unable to travel to a clinic initially.

Your First PT Session
Evaluation: assessment of strength, range of motion, balance, gait, and functional abilities. Goal setting: establishing short-term (2–4 week) and long-term (3–6 month) goals. Initial exercises: gentle, progressive, focused on pain management, swelling control, and early mobility.

Common PT Modalities
Therapeutic exercise (primary modality). Manual therapy: hands-on techniques for joint mobilization, soft tissue release. Electrical stimulation (TENS, NMES): pain management and muscle activation. Ultrasound therapy: deep tissue healing. Ice and heat modalities. Taping and bracing.

Home Exercise Programs (HEP)
Your PT will prescribe home exercises to perform daily between sessions. Consistency with HEP is equally important as clinic attendance. Keep an exercise log.

How Long Will PT Last?
Minor surgery: 4–8 weeks of PT. Joint replacement: 8–12 weeks. Spinal fusion: 12–24 weeks. Complex procedures or slow progress: longer. Home maintenance program should continue indefinitely after formal PT concludes.

Insurance and Cost
Most private insurance plans cover post-operative PT with a referral. Medicare covers PT up to a benefit cap per year. Verify your coverage before beginning treatment.
""",

    "doc_46_range_of_motion_exercises.txt": """
RANGE OF MOTION EXERCISES AFTER SURGERY

What Is Range of Motion (ROM)?
Range of motion refers to the degree of movement possible at a joint. Surgery, inflammation, and immobility reduce ROM. Restoring full or near-full ROM is a primary goal of post-operative rehabilitation for joint and orthopedic surgeries.

Types of ROM Exercises
Passive ROM (PROM): the limb is moved by a therapist, caregiver, or machine — no muscle effort by patient. Used in the early, most acute phase when active muscle use is restricted. Active-assisted ROM (AAROM): patient initiates movement with assistance from therapist or the other limb. Active ROM (AROM): patient moves the joint independently through available range.

Continuous Passive Motion (CPM) Machine
After knee replacement, a CPM machine may be prescribed. It gently flexes and extends the knee continuously while you rest. Modern evidence shows mixed benefit; use per your surgeon's specific protocol. Typical settings: 0–90° flexion range, 2–4 hours daily.

Sample ROM Exercises After Knee Surgery
Heel slides: lying on back, slowly slide heel toward buttocks, bending knee as far as comfortable. Hold 5 seconds. Return. Quad sets: lying on back, tighten thigh muscle, press back of knee toward bed. Hold 10 seconds. Ankle pumps: point toes toward ceiling, then away. Repeat 20 times per hour (prevents blood clots).

Sample ROM Exercises After Shoulder Surgery
Pendulum exercise: lean forward, let arm hang, gently swing arm in small circles. Passive forward flexion: therapist or other hand slowly raises arm forward as far as comfortable.

Progression and Targets
Set specific ROM targets with your PT (e.g., knee flexion of 90° by 4 weeks, 120° by 8 weeks). Progress slowly — pain with exercise should be mild and temporary. Sharp or severe pain signals stopping.
""",

    "doc_47_strength_training_recovery.txt": """
REBUILDING STRENGTH AFTER SURGERY

Why Muscle Strength Decreases After Surgery
Significant muscle atrophy (loss of muscle mass) begins within 24–48 hours of immobility. The combination of disuse, protein catabolism from surgical stress, and pain-related guarding causes rapid strength loss. Rebuilding strength is essential for: functional independence, joint stability, return to work and sports, and long-term quality of life.

Phases of Strength Recovery
Phase 1 (Weeks 1–4): Muscle activation — focuses on waking up muscles, preventing further atrophy. Isometric exercises (muscle contraction without joint movement): quad sets, glute sets, abdominal bracing. Phase 2 (Weeks 4–8): Strengthening — progressive resistance exercises with body weight, resistance bands, or light weights. Phase 3 (Weeks 8–16): Return to function — sport or occupation-specific strength training, balance challenges. Phase 4 (3–12 months): Return to full activity with maintenance of strength gains.

Principles of Strength Training in Recovery
Progressive overload: gradually increase resistance as strength improves. Rest and recovery: exercise every other day to allow muscle repair. Pain as a guide: mild muscle soreness (DOMS) is acceptable; joint pain or sharp pain is not. Proper form: always prioritize form over amount of weight to prevent re-injury.

Common Post-Surgical Strengthening Exercises
Straight leg raises (SLR): lying on back, contract quad, lift straight leg to 45°. Terminal knee extensions (TKE): strengthen quad in last 30° of extension. Clamshells: hip abductor/external rotator strengthening. Heel raises: calf strengthening, single-leg when possible. Bridges: gluteal and core strengthening.

Role of Protein in Muscle Rebuilding
Strength training without adequate protein intake yields poor results. Target 1.2–1.5 g/kg/day protein. Consume protein within 30 minutes of strength training to maximize muscle protein synthesis.
""",

    "doc_48_aquatic_therapy.txt": """
AQUATIC THERAPY IN SURGICAL RECOVERY

Overview of Aquatic Therapy
Aquatic therapy (hydrotherapy) uses the properties of water — buoyancy, resistance, hydrostatic pressure, and temperature — to facilitate rehabilitation in a low-impact environment. It is especially beneficial for patients whose weight-bearing on land is painful or restricted due to orthopedic surgery.

When Can Aquatic Therapy Begin?
Aquatic therapy requires fully healed surgical wounds (no open areas, no active drainage) — typically 4–6 weeks after surgery for most procedures. Swimming, pool exercises, and underwater treadmill may all be used.

Benefits of Aquatic Therapy
Buoyancy reduces body weight: at chest depth, approximately 70–75% of body weight is supported, allowing pain-free movement that is not possible on land. Hydrostatic pressure reduces swelling. Water resistance (8–15 times air resistance) strengthens muscles without heavy loading. Warm water reduces pain and muscle spasm. Improved proprioception and balance in a safe, fall-minimized environment.

Who Benefits Most?
Total hip and knee replacement patients: begin walking in the pool earlier than on land. Spinal surgery patients: axial loading reduced, allowing earlier movement. Obese patients: significantly reduced joint loading. Elderly patients: low fall risk in water compared to land exercises.

Precautions
Do not enter pool until wound is fully healed. Monitor for signs of infection after pool sessions. Avoid pool if urinary tract infection, open skin lesions, or active infection is present. First sessions should be supervised by a physical therapist.

Sample Pool Exercises
Pool walking forward and backward. Side-stepping in water. Hip flexion and extension while holding pool wall. Knee bends (squats) in chest-deep water. Arm sweeping exercises.
""",

    "doc_49_occupational_therapy.txt": """
OCCUPATIONAL THERAPY AFTER SURGERY

What Is Occupational Therapy?
Occupational therapy (OT) focuses on restoring the ability to perform activities of daily living (ADLs) — tasks such as dressing, bathing, grooming, preparing meals, and returning to work or leisure activities. After surgery, OT bridges the gap between medical recovery and functional independence.

When Is OT Recommended?
OT is particularly valuable after: upper extremity surgery (hand, wrist, shoulder, elbow), total joint replacement (hip, knee), stroke or brain surgery, spinal surgery with neurological effects, and surgery affecting self-care abilities.

ADL Assessment and Training
Your occupational therapist will assess your ability to perform: personal hygiene and bathing, dressing (upper and lower body), meal preparation, home management tasks, work-related activities. Adaptive techniques and equipment are introduced when standard methods are not safe.

Adaptive Equipment
Long-handled reacher: pick up objects from the floor without bending (after hip surgery). Long-handled shoehorn and sock aid: don socks and shoes without bending over. Raised toilet seat: maintain hip precautions after hip replacement. Grab bars in shower and bathroom: prevent falls. Buttonhook and zipper pull: for fine motor deficits after hand surgery.

Return to Work
OT can perform functional capacity evaluations (FCE) to assess readiness to return to specific job tasks. Recommendations for ergonomic workstation modifications, activity pacing, and gradual return-to-work schedules are provided.

Hand Therapy (Subspecialty OT)
Certified Hand Therapists (CHTs) are OTs or PTs specializing in upper extremity rehabilitation. They address: post-surgical edema, scar management, range of motion and strengthening, custom splinting, and desensitization.
""",

    "doc_50_managing_fatigue.txt": """
MANAGING FATIGUE DURING SURGICAL RECOVERY

Why Surgical Fatigue Is Real
Post-operative fatigue is one of the most common and undertreated aspects of surgical recovery. It has multiple causes: the body's enormous energy expenditure for wound healing and immune defense, anesthesia effects, blood loss, poor sleep, pain, nutritional deficits, and psychological stress. It is not a sign of weakness — it is a physiological consequence of the surgical process.

How Long Does Fatigue Last?
Fatigue peaks in the first 1–2 weeks and gradually improves. After minor surgery: 2–4 weeks. After major surgery (joint replacement, cardiac, abdominal): 4–12 weeks. Full energy restoration may take 3–6 months.

Energy Conservation Strategies
Plan and prioritize: identify the most important activities for the day and do them when your energy is highest (usually morning). Rest before and after activity, not just when you are exhausted. Pace yourself: alternate periods of activity with short rest periods. Do not push through severe fatigue.

Sleep Is Healing
Prioritize sleep. Aim for 8–9 hours nightly. Nap if needed during the day (30 minutes is ideal; avoid naps after 3 PM to protect nighttime sleep).

Nutrition for Energy
Fatigue is worsened by inadequate calorie or protein intake. Even if appetite is poor, eat regular small meals. Include protein in every meal. Iron deficiency anemia (common after surgical blood loss) causes profound fatigue — your doctor will check your hemoglobin level; iron supplements or infusions may be needed.

When Fatigue May Indicate a Problem
Fatigue gradually worsening after initial improvement. Severe fatigue accompanied by: shortness of breath, rapid heart rate, pallor — possible anemia. Fatigue with fever — possible infection. Depression-related fatigue: see mental health section.

Exercise Paradox
Gentle progressive exercise (walking) paradoxically improves energy levels. Start with short walks and gradually increase — rest does not cure fatigue; activity (in appropriate amounts) does.
""",

    "doc_51_mental_health_after_surgery.txt": """
MENTAL HEALTH AND EMOTIONAL WELL-BEING AFTER SURGERY

The Emotional Impact of Surgery
Surgery is not only a physical event — it carries significant psychological weight. Patients commonly experience anxiety, depression, frustration, fear, and loss of independence during recovery. Acknowledging and addressing these emotional responses is essential to holistic recovery.

Pre-Operative Anxiety
Anxiety before surgery is universal. Techniques that help: education about the procedure, meeting with the anesthesiologist in advance, breathing and relaxation exercises, and honest communication with the surgical team about concerns.

Post-Operative Depression
Mild mood disturbance (tearfulness, irritability, low mood) is very common in the first 1–2 weeks. Contributing factors: pain, poor sleep, loss of normal routine, reduced independence, anesthesia effects, and physiological stress response. This typically resolves as recovery progresses.

When to Seek Help for Depression
If low mood persists beyond 2–3 weeks; if you lose interest in activities you normally enjoy; if you experience persistent hopelessness, worthlessness, or suicidal thoughts — seek professional help immediately. Call a mental health professional or your primary care doctor. In the US, the 988 Suicide and Crisis Lifeline is available 24/7.

Post-Operative Cognitive Dysfunction (POCD)
POCD — temporary cognitive impairment including memory problems, difficulty concentrating, and slowed thinking — occurs in up to 20% of elderly patients after major surgery. It is usually transient, resolving within weeks to months. Minimize opioid use and sleep disturbance to reduce POCD risk.

Anxiety About Surgical Outcome
Some patients experience intense anxiety about whether the surgery was successful or if they are healing correctly. Regular communication with your surgical team and follow-up appointments can address these fears. Cognitive behavioral therapy (CBT) is effective for health anxiety.

Support Systems
Social support from family, friends, and community is a strong predictor of recovery outcomes. Do not isolate yourself. Let people help. Consider joining a recovery support group for your procedure type.
""",

    "doc_52_sleep_hygiene_recovery.txt": """
SLEEP HYGIENE DURING SURGICAL RECOVERY

The Critical Role of Sleep in Healing
During deep sleep, the body produces growth hormone (GH), which is essential for tissue repair, protein synthesis, and immune function. Sleep deprivation profoundly impairs wound healing, immune defense, pain tolerance, and emotional regulation. Prioritizing sleep is not optional — it is medically necessary during surgical recovery.

Optimal Sleep Environment
Cool temperature: 65–68°F (18–20°C) is ideal for sleep. Dark room: use blackout curtains or a sleep mask. Quiet: use white noise or earplugs if ambient noise is disruptive. Comfortable support: extra pillows to support the surgical site in a comfortable, low-pain position.

Sleep Position After Surgery
Back surgery: lying flat or with knees slightly elevated on a pillow may reduce pain. Knee surgery: slight elevation of the leg reduces swelling and discomfort. Shoulder surgery: many patients find sleeping in a recliner or propped upright more comfortable. Hip replacement: wedge pillow between legs maintains hip precautions.

Sleep Schedule Consistency
Go to bed and wake up at the same time every day — even on weekends and holidays. Consistency regulates the circadian rhythm. Avoid long daytime naps (limit to 20–30 minutes and avoid after 3 PM).

Light and Circadian Rhythm
Exposure to bright light (natural sunlight or light therapy box) in the morning helps reset circadian rhythm disrupted by hospitalization. Avoid screens (phone, TV) 1 hour before bed — blue light suppresses melatonin production.

Managing Pain at Night
Take pain medications approximately 30–45 minutes before bedtime to ensure they are working at their peak when you try to sleep. Ice the surgical site before bed to reduce nighttime discomfort. Use a pillow barrier to prevent accidental rolling onto the surgical site.
""",

    "doc_53_nutrition_healing.txt": """
NUTRITION FOR OPTIMAL WOUND HEALING

Overview
The body requires significantly increased nutritional support after surgery to fuel the complex biological processes of wound healing: inflammation, cellular proliferation, collagen synthesis, tissue remodeling, and immune defense. Inadequate nutrition is a leading cause of delayed healing, wound complications, and extended recovery.

Caloric Requirements
Total caloric needs increase 15–30% after major surgery. For a typical adult: 2000–2500 kcal/day for minor surgery; 2500–3000+ kcal/day for major surgery or trauma. Avoid extreme caloric restriction during recovery — it impairs healing even in obese patients.

The Healing Nutrients

Protein (most important nutrient)
Role: building block for collagen, immune cells, enzymes, and all cellular repair. Requirement: 1.2–2.0 g/kg/day (higher for major surgery, burns, or multiple procedures). Sources: eggs, lean meat, fish, poultry, Greek yogurt, legumes, whey protein supplements.

Vitamin C
Role: essential cofactor for collagen synthesis and antioxidant protection. Sources: citrus fruits, bell peppers, kiwi, strawberries. Supplemental dose: 500–1000 mg/day during healing.

Zinc
Role: required for protein synthesis, wound contraction, and immune function. Sources: red meat, shellfish (especially oysters), pumpkin seeds, nuts. Supplemental dose: 25–45 mg/day for 8–12 weeks.

Copper
Role: cofactor for cross-linking collagen and elastin. Often depleted with high-dose zinc supplementation (zinc inhibits copper absorption). Sources: organ meats, shellfish, nuts, whole grains.

Iron
Role: required for oxygen transport (hemoglobin) and hydroxylation of proline in collagen. Sources: red meat, spinach, fortified cereals. Note: iron absorption is enhanced by vitamin C and inhibited by calcium and tea.

Hydration
Adequate hydration is critical for circulation, oxygen delivery to tissues, and flushing metabolic waste. Aim for 8–10 cups of water daily unless fluid restricted.
""",

    "doc_54_hydration_guide.txt": """
HYDRATION AFTER SURGERY

Why Hydration Matters After Surgery
Surgical patients are often dehydrated for multiple reasons: preoperative fasting (NPO), blood loss during surgery, increased fluid loss through fever and wound drainage, reduced fluid intake post-operatively due to nausea, and fluid shifts from tissues into the "third space" (body cavities). Dehydration after surgery impairs wound healing, kidney function, medication processing, and cardiovascular stability.

How Much to Drink
General target: 8–10 cups (64–80 oz / 2–2.5 liters) of water daily for most patients. Patients with heart failure, kidney disease, or fluid retention: follow your doctor's specific fluid restriction. Patients with drains, high output, or fever: increase intake accordingly.

Signs of Adequate Hydration
Urine color: pale yellow (like lemonade). Urinating every 2–4 hours. No significant thirst, dry mouth, or dizziness.

Signs of Dehydration
Dark yellow or amber urine. Decreased urination (less than 4 times per day). Dry mouth and lips. Headache and dizziness. Rapid heart rate. Constipation. Fatigue and confusion.

Best Fluids to Drink
Water: best choice. Dilute fruit juices and sports drinks: provide electrolytes (especially relevant if significant sweating or fever). Broth: provides sodium and can settle the stomach. Herbal teas, decaf coffee: count toward fluid intake. Avoid: alcoholic beverages (diuretic effect), high-sugar sodas, energy drinks.

IV Fluids in the Hospital
In the hospital, IV fluids (normal saline, lactated Ringer's solution) maintain hydration when oral intake is restricted. Excessive IV fluids can cause fluid overload — your medical team carefully monitors your fluid balance.

When to Seek Help
Unable to keep fluids down for more than 6–8 hours due to vomiting. Producing less than 500 mL of urine per day. Severe dizziness upon standing. Confusion or extreme weakness.
""",

    "doc_55_smoking_cessation_surgery.txt": """
SMOKING, SURGERY, AND RECOVERY

The Impact of Smoking on Surgical Recovery
Smoking is one of the most significant modifiable risk factors for post-operative complications. Smokers have substantially higher rates of: wound healing complications (incisions heal more slowly and are more prone to infection), pulmonary complications (pneumonia, prolonged intubation), cardiovascular events (heart attack, blood clot), anastomotic leaks (after bowel surgery), and implant failure (higher rates of infection and loosening after joint replacement).

How Smoking Impairs Healing
Nicotine causes vasoconstriction (narrowing of blood vessels), reducing blood and oxygen delivery to healing tissues. Carbon monoxide from tobacco smoke displaces oxygen from hemoglobin, further reducing tissue oxygen levels. Wound oxygenation is a critical determinant of healing — smoking can reduce wound oxygen tension by 40%.

Perioperative Smoking Cessation
Stopping smoking at least 4–8 weeks before surgery significantly reduces complication rates. Even stopping 2–4 weeks before surgery provides measurable benefits. Stopping the night before surgery offers some protection against pulmonary complications.

Nicotine Replacement and Cessation Support
Options: nicotine patch, nicotine gum, nicotine lozette, varenicline (Chantix), bupropion (Wellbutrin). Note: some surgeons restrict nicotine replacement (patches) around the time of surgery — confirm with your surgical team. Behavioral support: counseling, quitline (1-800-QUIT-NOW), support apps (Quit Genius, Smoke Free).

After Surgery
Many patients find surgery is a strong motivating event for permanent smoking cessation. Use the recovery period to build on cessation momentum. Studies show patients who quit smoking around surgery have higher long-term quit rates.
""",

    "doc_56_diabetes_management_surgery.txt": """
DIABETES MANAGEMENT DURING SURGICAL RECOVERY

Why Diabetes Complicates Surgery and Recovery
Diabetes mellitus, both type 1 and type 2, significantly increases surgical risk and complication rates: impaired immune function increases infection risk, impaired microvascular circulation reduces wound healing and oxygen delivery, poor glucose control (high blood sugar) inhibits white blood cell function, and diabetes-related neuropathy may mask pain and infection warning signs.

Perioperative Blood Glucose Targets
Target blood glucose: 140–180 mg/dL (7.8–10 mmol/L) in the hospital. Avoid hypoglycemia (< 70 mg/dL / 3.9 mmol/L) — also dangerous post-operatively. Tight control (below 140 mg/dL) in the ICU has been shown to improve outcomes in some studies; generally hospitals target the 140–180 range to balance benefit and hypoglycemia risk.

Insulin Management Around Surgery
Basal insulin (long-acting): typically continued at 75–80% of home dose the night before and morning of surgery. Bolus insulin (short-acting): held until patient is eating. Oral diabetes medications: metformin is typically held 24–48 hours before and after surgery due to lactic acidosis risk; other medications per anesthesia team guidance.

Blood Glucose Monitoring Post-Surgery
Monitor blood glucose every 4–6 hours initially. Adjust insulin accordingly. Stress-induced hyperglycemia (high blood sugar from surgical stress) occurs even in non-diabetics — more pronounced in diabetics. High blood sugar (> 180 mg/dL) persistently post-operatively predicts higher infection rates.

Wound Healing in Diabetics
Enhanced wound care protocols for diabetic patients. Inspect surgical sites daily. Report any signs of infection (redness, warmth, drainage) immediately. Adequate glucose control is the single most important wound-healing intervention for diabetics.
""",

    "doc_57_elderly_surgery_recovery.txt": """
SPECIAL CONSIDERATIONS FOR ELDERLY PATIENTS AFTER SURGERY

Physiological Changes in the Elderly Affecting Recovery
Aging affects every organ system, making surgery and recovery more complex: reduced cardiovascular reserve (heart and lungs have less capacity to respond to stress), decreased kidney function (slower drug clearance, fluid sensitivity), reduced muscle mass (sarcopenia — greater pre-existing weakness), impaired immune function, reduced bone density, slower wound healing, and altered drug metabolism (greater sensitivity to medications).

Delirium — A Common and Serious Complication
Post-operative delirium (acute confusion) affects 10–50% of elderly patients after major surgery. It presents as fluctuating confusion, disorientation, agitation, or unusual behavior. Prevention: adequate sleep, frequent reorientation, early mobilization, hearing aids and glasses in place, pain control, avoidance of Foley catheters, and minimizing sedating medications. Alert the medical team immediately if confusion develops.

Falls Prevention
Elderly patients are at significantly higher fall risk after surgery: pain medications cause dizziness, muscle weakness from bed rest, and orthostatic hypotension. Measures: call for assistance before getting up, use a walker or cane as prescribed, keep the path clear of trip hazards at home, install grab bars in the bathroom.

Medication Sensitivity
The "Beers Criteria" identifies medications that are high-risk for older adults: diphenhydramine (Benadryl) causes increased confusion; certain muscle relaxants cause excessive sedation; benzodiazepines increase fall and delirium risk; opioids require careful dosing (start lower, go slower).

Frailty and Prehabilitation
Frail elderly patients benefit from prehabilitation (pre-operative optimization): nutritional supplementation, pre-operative exercise program, and medical optimization to reduce post-operative complication risk. A frailty assessment should ideally be performed before elective surgery in patients over 70.
""",

    "doc_58_pediatric_surgery_recovery.txt": """
RECOVERY FROM SURGERY IN CHILDREN

Pain Assessment in Children
Children experience pain from surgery and require effective pain management. Pain assessment tools vary by age: FLACC scale (Face, Legs, Activity, Cry, Consolability) for infants and pre-verbal children. Wong-Baker FACES scale: for children 3 years and older. Numeric scale (0–10): for children 8 years and older who understand numbers.

Managing Pain in Children Post-Surgery
Acetaminophen (Tylenol) and ibuprofen are the cornerstones of pediatric post-operative pain management. Opioids are prescribed for significant pain and administered carefully. Children often benefit from non-pharmacological pain strategies: distraction (toys, screens, music), physical contact (holding, rocking), sucrose solution for infants.

Wound Care for Children
Younger children may pick at incisions — protect with comfortable dressings and distraction. Steri-strips fall off naturally; do not force removal. Age-appropriate explanation of the wound and healing helps reduce anxiety. For water activities, keep incisions covered as directed.

Eating and Drinking After Pediatric Surgery
Encourage fluid intake first. Popsicles, ice chips, and clear fluids are often well-tolerated. Advance diet as tolerated; nausea is common after general anesthesia. Never force eating.

Activity and School Return
Children recover faster than adults but also tend to overdo activity — parents must monitor. Specific activity restrictions depend on the surgery type and are provided by the surgeon. School return: typically 1–2 weeks for minor procedures; longer for major surgery. PE class: usually restricted for 2–6 weeks depending on procedure.

Emotional Support for Pediatric Patients
Hospital stays and surgery can be traumatic for children. Post-hospitalization behavior changes (nightmares, clinginess, regression) are common and usually temporary. Consistent reassurance, normal routine restoration, and gentle explanation of events support adjustment.
""",

    "doc_59_returning_to_exercise.txt": """
RETURNING TO EXERCISE AFTER SURGERY

Why Exercise Matters for Long-Term Recovery
Regular physical activity after surgical recovery: maintains weight loss (especially relevant after bariatric surgery), preserves and builds muscle mass rebuilt during recovery, reduces risk of cardiovascular disease, improves mood, sleep quality, and bone density, and prevents de-conditioning (loss of fitness).

General Guidelines for Exercise Resumption
The timeline for returning to exercise varies widely by procedure: Minor surgery (hernia repair, appendectomy): light exercise at 2–4 weeks; unrestricted at 6 weeks. Moderate surgery (knee arthroscopy, abdominal laparoscopy): 4–8 weeks. Major surgery (joint replacement, open abdominal, cardiac): 6–12+ weeks.

Always obtain explicit clearance from your surgeon before resuming structured exercise.

Progression Principles
Start low, go slow: begin with light walking and gradually increase intensity. Never increase mileage or intensity by more than 10% per week (the "10% rule") to avoid overuse injuries. Pain is a guide: mild muscular discomfort is acceptable; joint pain, wound pain, or any unusual pain is a signal to stop and reassess.

Recommended Exercise Types During Recovery
Low-impact cardio: walking, cycling (stationary initially), swimming (once wound healed), elliptical trainer. Strength training: progressive resistance training under guidance of physical therapist or certified trainer familiar with your surgery. Flexibility: gentle stretching to maintain range of motion.

Exercises to Avoid After Specific Surgeries
After spine fusion: no heavy axial loading, no twisting/bending until cleared. After joint replacement: no high-impact activities (running, jumping) long-term. After hernia repair: avoid Valsalva-type exercises (straining under heavy load) that raise intra-abdominal pressure.

Exercise and Mental Health
Exercise has powerful antidepressant and anxiolytic effects. The return to physical activity is often associated with improved mood, confidence, and sense of normalcy in post-surgical patients.
""",

    "doc_60_caregiver_guide.txt": """
GUIDE FOR CAREGIVERS OF POST-SURGICAL PATIENTS

The Vital Role of the Caregiver
Family members, partners, and friends who provide care during surgical recovery are invaluable. Research consistently shows that patients with strong social support recover faster, have fewer complications, and experience better emotional well-being. However, caregiving is also physically and emotionally demanding — caregiver self-care is essential.

What Caregivers Do
Practical support: transportation to/from the hospital and follow-up appointments, medication management (picking up prescriptions, organizing medication schedules), meal preparation, grocery shopping, wound care assistance, assistance with personal hygiene. Monitoring: tracking symptoms, vital signs (temperature, pain score), drain output, and knowing when to call for help. Emotional support: listening, reassurance, companionship, and normalizing recovery challenges.

Understanding the Recovery Timeline
Caregivers benefit from knowing the expected recovery timeline and milestones for the specific surgery. This helps set realistic expectations and reduces caregiver alarm over normal recovery events (bruising, fatigue, temporary mood changes).

Medication Management
Keep a complete medication list with doses and schedules. Set phone alarms for medication times. Organize medications in a weekly pill organizer. Be aware of drug interactions and signs of adverse effects. Never administer a medication not prescribed for your loved one.

Red Flags — When to Call 911 or Go to the ER
Chest pain, severe shortness of breath, loss of consciousness → Call 911. Signs of pulmonary embolism (one-sided leg swelling with pain AND shortness of breath). Signs of stroke (sudden face drooping, arm weakness, speech difficulty). Heavy uncontrolled bleeding. Signs of severe infection with rapid deterioration.

Caregiver Self-Care
Caregiver burnout is real. Accept help from others. Maintain your own sleep, exercise, and nutrition. Identify and use community resources (home health aides, meals programs). Communicate honestly with the surgical team if caregiving needs exceed your capacity.
""",
}


def generate_corpus():
    """Write all documents to the data/documents directory."""
    count = 0
    for filename, content in DOCUMENTS.items():
        filepath = DOCS_DIR / filename
        filepath.write_text(content.strip(), encoding="utf-8")
        count += 1
    print(f"Generated {count} documents in {DOCS_DIR}")


if __name__ == "__main__":
    generate_corpus()
