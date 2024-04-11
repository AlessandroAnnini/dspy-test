#####################
# Accounting Categories Advisor
#####################

import os
import dspy
from dspy.teleprompt import LabeledFewShot, BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Set up the LM
gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=300, api_key=OPENAI_API_KEY)
dspy.configure(lm=gpt3_turbo)


#####################
# Building Signatures
#####################


class GenerateAnswer(dspy.Signature):
    """Studia il seguente conto economico ed assegna la corretta categoria contabile (category) di conto economico all'elemento fornito (line_item).

    # Conto Economico Azienda Tessile

    ## Ricavi
    - **Ricavi delle vendite:** Vendite di tessuti, abbigliamento e accessori prodotti.
    - **Ricavi da servizi:** Entrate da servizi di design, consulenza, e post-vendita.

    ## Costi della Produzione
    - **Costi delle materie prime:** Acquisto di filati, tessuti grezzi, e altri materiali.
    - **Costi del lavoro diretto:** Salari del personale di produzione.
    - **Costi di produzione indiretti:** Ammortamento delle attrezzature, manutenzione, energia elettrica.

    ## Valore della Produzione
    - **Totale Ricavi**
    - **Meno: Costi della Produzione**

    ## Risultato Operativo Lordo
    - **Margine Operativo Lordo:** Differenza tra il valore della produzione e i costi della produzione.

    ## Spese Operative
    - **Spese Amministrative:** Salari del personale amministrativo, forniture d'ufficio.
    - **Spese di Vendita:** Pubblicità, commissioni ai venditori, spese di trasporto e distribuzione.
    - **Ricerca e Sviluppo:** Costi relativi allo sviluppo di nuovi prodotti o miglioramento dei prodotti esistenti.
    - **Ammortamenti:** Ammortamento degli impianti e delle attrezzature.

    ## Risultato Operativo Netto
    - **Margine Operativo Netto:** Differenza tra il risultato operativo lordo e le spese operative.

    ## Proventi e Oneri Finanziari
    - **Interessi Passivi:** Interessi su prestiti e finanziamenti.
    - **Interessi Attivi:** Interessi maturati su investimenti e depositi.

    ## Risultato Prima delle Imposte
    - **Utile o Perdita Operativa:** Differenza tra il risultato operativo netto e i proventi e oneri finanziari.

    ## Imposte sul Reddito
    - **Imposte sul reddito delle società**

    ## Utile (Perdita) Netto/a
    - **Utile o Perdita Netto/a:** Differenza tra il risultato prima delle imposte e le imposte sul reddito.
    """

    line_item = dspy.InputField(desc="L'elemento da categorizzare.")
    answer = dspy.OutputField(desc="La giusta categoria di conto economico.")


#####################
# Building the Pipeline
#####################


class AccountingCategoryAdvisor(dspy.Module):
    def __init__(self):
        super().__init__()

        self.generate_answer = dspy.Predict(GenerateAnswer)

    def forward(self, line_item):
        prediction = self.generate_answer(line_item=line_item)
        return dspy.Prediction(answer=prediction.answer)


#####################
# Using AI feedback for the metric
#####################


# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the correctness of a classification task."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")


def metric(gold, pred, trace=None):
    line_item, category, answer = gold.line_item, gold.category, pred.answer

    correct = f"The answer should classify the line item '{line_item}' in the category '{category}'."

    with dspy.context(lm=gpt3_turbo):
        correct = dspy.Predict(Assess)(
            assessed_text=answer, assessment_question=correct
        )

    correct = correct.assessment_answer.lower() == "yes"
    score = 1 if correct else 0

    if trace is not None:
        return score >= 1
    return score


#####################
# Building the datasets
#####################

traindata = [
    {
        "line_item": "Ricevuta energia termica",
        "category": "Costi di produzione indiretti",
    },
    {
        "line_item": "Fattura vendita abbigliamento finito",
        "category": "Ricavi delle vendite",
    },
    {"line_item": "Nota di credito per reso merci", "category": "Ricavi delle vendite"},
    {"line_item": "Pagamento royalty design", "category": "Spese Amministrative"},
    {
        "line_item": "Fattura servizio di pulizia uffici",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Ricevuta acquisto licenza software gestionale",
        "category": "Spese Amministrative",
    },
    {"line_item": "Bolletta gas fabbrica", "category": "Costi di produzione indiretti"},
    {
        "line_item": "Fattura acquisto filati speciali",
        "category": "Costi delle materie prime",
    },
    {
        "line_item": "Pagamento contributi previdenziali dipendenti",
        "category": "Costi del lavoro diretto",
    },
    {
        "line_item": "Ricevuta noleggio automezzi per distribuzione",
        "category": "Spese di Vendita",
    },
    {
        "line_item": "Fattura acquisto software CAD per design tessuti",
        "category": "Ricerca e Sviluppo",
    },
    {
        "line_item": "Pagamento interessi su obbligazioni",
        "category": "Interessi Passivi",
    },
    {"line_item": "Entrate da brevetti", "category": "Altri ricavi e redditi"},
    {
        "line_item": "Fattura servizi di consulenza legale",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Pagamento tasse su proprietà industriale",
        "category": "Spese Amministrative",
    },
    {"line_item": "Bonifico per ristrutturazione impianti", "category": "Ammortamenti"},
    {
        "line_item": "Ricevuta manutenzione straordinaria macchinari",
        "category": "Costi di produzione indiretti",
    },
    {"line_item": "Fattura servizi di trasporto merci", "category": "Spese di Vendita"},
    {
        "line_item": "Pagamento assicurazione fabbrica",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Ricevuta acquisto uniformi per il personale",
        "category": "Costi del lavoro diretto",
    },
    {"line_item": "Fattura stampa cataloghi prodotti", "category": "Spese di Vendita"},
    {
        "line_item": "Pagamento servizio hosting sito web",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Fattura per acquisto stand fieristici",
        "category": "Spese di Vendita",
    },
    {"line_item": "Ricevuta commissioni su vendite", "category": "Spese di Vendita"},
    {
        "line_item": "Fattura per servizi fotografici prodotti",
        "category": "Spese di Vendita",
    },
    {
        "line_item": "Bonifico per investimenti in sicurezza sul lavoro",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Ricevuta pagamento utenze ufficio",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Fattura acquisto tessuti per campionature",
        "category": "Costi delle materie prime",
    },
    {
        "line_item": "Ricevuta spese di viaggio per fiere",
        "category": "Spese di Vendita",
    },
    {
        "line_item": "Pagamento servizi postali e di corriere",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Bonifico per acquisto quote di mercato",
        "category": "Altri ricavi e redditi",
    },
    {
        "line_item": "Fattura per servizi di market analysis",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Pagamento canoni di leasing per automezzi",
        "category": "Costi di produzione indiretti",
    },
    {
        "line_item": "Ricevuta acquisto materiale promozionale",
        "category": "Spese di Vendita",
    },
    {
        "line_item": "Fattura energia elettrica showroom",
        "category": "Costi di produzione indiretti",
    },
    {
        "line_item": "Pagamento servizio di recupero crediti",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Fattura acquisto attrezzature per qualità",
        "category": "Costi di produzione indiretti",
    },
    {
        "line_item": "Bonifico per formazione personale",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Ricevuta per servizi SEO sul sito aziendale",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Pagamento diritti d'autore su design",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Fattura per consulenza finanziaria",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Ricevuta bonifico per acquisto brevetti",
        "category": "Altri ricavi e redditi",
    },
    {
        "line_item": "Pagamento interessi su mutuo fabbrica",
        "category": "Interessi Passivi",
    },
    {
        "line_item": "Fattura per acquisto computer ufficio",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Ricevuta abbonamento riviste settoriali",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Bonifico per investimenti pubblicitari online",
        "category": "Spese di Vendita",
    },
    {
        "line_item": "Fattura per rinnovo licenze software",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Pagamento quota annuale associazione categoria",
        "category": "Spese Amministrative",
    },
    {
        "line_item": "Ricevuta acquisto gadget promozionali",
        "category": "Spese di Vendita",
    },
    {
        "line_item": "Bonifico per aggiornamento impianti di sicurezza",
        "category": "Costi di produzione indiretti",
    },
]

trainset = [
    dspy.Example(line_item=e["line_item"], category=e["category"]).with_inputs(
        "line_item"
    )
    for e in traindata
]

testdata = [
    {"line_item": "Fattura di vendita tessuti", "category": "Ricavi delle vendite"},
    {"line_item": "Ricevuta acquisto cotone", "category": "Costi delle materie prime"},
    {
        "line_item": "Bolletta elettrica fabbrica",
        "category": "Costi di produzione indiretti",
    },
    {"line_item": "Stipendi operai tessitura", "category": "Costi del lavoro diretto"},
    {"line_item": "Nota spese viaggio commerciale", "category": "Spese di Vendita"},
    {"line_item": "Fattura acquisto macchinario tessile", "category": "Ammortamenti"},
    {
        "line_item": "Estratto conto interessi passivi prestito",
        "category": "Interessi Passivi",
    },
    {"line_item": "Ricevuta pagamento pubblicità", "category": "Spese di Vendita"},
    {"line_item": "Fattura consulenza design moda", "category": "Spese Amministrative"},
    {
        "line_item": "Bonifico per investimento in ricerca",
        "category": "Ricerca e Sviluppo",
    },
]

testset = [
    dspy.Example(line_item=e["line_item"], category=e["category"]).with_inputs(
        "line_item"
    )
    for e in testdata
]

# trainset = trainset.with_inputs("line_item").with_outputs("category")

#####################
# Optimize and compile the Model
#####################

# Define teleprompter
# teleprompter = LabeledFewShot()

teleprompter = BootstrapFewShot(metric=metric)

# Compile!
compiled_ACA = teleprompter.compile(
    student=AccountingCategoryAdvisor(), trainset=trainset
)

#####################
# Evaluate the Model
#####################


# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(
    devset=testset,
    metric=metric,
    num_threads=4,
    display_progress=True,
    display_table=5,
)


uncompiled_evaluation = evaluate(AccountingCategoryAdvisor(), metric)
compiled_evaluation = evaluate(compiled_ACA, metric)

print(f"## Score for uncompiled: {uncompiled_evaluation}")
print(f"## Score for compiled: {compiled_evaluation}")

#####################
# Make Predictions
#####################

line_item_question = "Bolletta elettrica fabbrica"

# Get the prediction. This contains `pred.context` and `pred.answer`.
pred = compiled_ACA(line_item_question)

print(f"Question: {line_item_question}")
print(f"Predicted Answer: {pred.answer}")


score = 0
for test in testset:
    line_item = test["line_item"]
    category = test["category"]

    pred = compiled_ACA(line_item)
    if pred.answer == category:
        score += 1

    print(
        f"Question: {line_item}, Predicted Answer: {pred.answer}, Expected Answer: {category}, Score: {score}"
    )
    print("\n")

print(f"Final Score: {score}/{len(testset)}")
