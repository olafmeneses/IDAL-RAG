from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

### Router
router_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Eres un experto en dirigir una 
    pregunta de un usuario a una vectorstore o a una búsqueda web. Usa la vectorstore para preguntas sobre {vectorstore_topics}. 
    No es necesario ser estricto con las palabras clave en la pregunta relacionada con estos temas. 
    De lo contrario, utiliza la búsqueda web. Da una opción binaria 'web_search' o 'vectorstore' en función de la pregunta. 
    Devuelve un JSON con una sola clave 'datasource' y sin preámbulos o explicaciones. 
    Pregunta que hay que dirigir: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "vectorstore_topics"],
)

### Rewriter
rewrite_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Eres un asistente que ayuda a generar 
    múltiples consultas de búsqueda basadas en una única consulta. Ten en cuenta que las consultas van a utilizarse
    para buscar información sobre archivos legales como la Constitución, Código Penal, etc.
    Genera {num_queries} consultas de búsqueda, relacionadas con la consulta inicial. Devuelve un JSON
    con una sola clave "queries" y una lista con las {num_queries} consultas generadas. 
    La lista debe contener las consultas y nada más.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Aquí está la consulta incial: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["num_queries", "question"],
)

### Retrieval Grader 
retrieval_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Eres un calificador que evalúa la relevancia 
    de un documento extraído con respecto a una pregunta del usuario. Si el documento contiene palabras clave relacionadas con la pregunta del usuario, 
    califícalo como relevante. No es necesario que sea una prueba rigurosa. El objetivo es filtrar las extracciones erróneas. \n
    Da una puntuación binaria 'yes' o 'no' para indicar si el documento es relevante para la pregunta. \n
    Proporciona la puntuación binaria como un JSON con una sola clave 'score' y sin preámbulos o explicaciones.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Aquí está el documento extraído: \n\n {document} \n\n
    Aquí está la pregunta del usuario: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

### Generate
gen_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Eres un asistente cuya tarea es responder preguntas basándose en extractos de documentos. 
    Usa los siguientes extractos de contexto recuperados para responder la pregunta. Si no sabes la respuesta o el contexto proporcionado es insuficiente, di que no sabes responderla.
    Intenta generar una respuesta concisa <|eot_id|><|start_header_id|>user<|end_header_id|>
    Pregunta: {question}
    Contexto: {context}
    Respuesta: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

### Generate raw
gen_raw_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Eres un asistente cuya tarea es responder preguntas.
    Intenta generar una respuesta concisa <|eot_id|><|start_header_id|>user<|end_header_id|>
    Pregunta: {question}
    Respuesta: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

### Hallucination Grader
hallucination_prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> Eres un calificador que evalúa si 
    una respuesta está fundamentada / respaldada por un conjunto de hechos. Da una puntuación binaria 'yes' o 'no' para indicar 
    si la respuesta está fundamentada / respaldada por un conjunto de hechos. Proporciona la puntuación binaria como un JSON con una  
    sola clave 'score' y sin preámbulos o explicaciones. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Aquí están los hechos:
    \n ------- \n
    {documents} 
    \n ------- \n
    Aquí está la respuesta: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

### Answer Eval
answer_eval_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Eres un calificador que evalúa la calidad de una respuesta de un modelo de lenguaje. \
    Da una puntuación entre 0 y 10 para indicar la calidad de la respuesta. Proporciona la puntuación como un JSON con una sola clave 'score' y sin preámbulos o explicaciones. \
    La respuesta verdadera a la pregunta es: {ground_answer}. \n \
    <|eot_id|><|start_header_id|>user<|end_header_id|> Aquí está la respuesta generada por el modelo:
    \n ------- \n
    {generation} 
    \n ------- \n
    Aquí está la pregunta: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question", "ground_answer"],
)
