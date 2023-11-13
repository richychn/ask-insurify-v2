from llama_index.vector_stores import VectorStoreQuery, PineconeVectorStore
from llama_index.schema import NodeWithScore
from llama_index.prompts import PromptTemplate
import pinecone 
import os
from dotenv import load_dotenv
load_dotenv()


class Query:
    def __init__(self, question, embedding, similarity_top_k=2, mode="default"):
        self.question = question
        self.embedding = embedding
        self.top_k = similarity_top_k
        self.mode = mode


        pinecone.init(api_key=os.getenv("PINECONE_API"), environment=os.getenv("PINECONE_ENV"))
        self.pinecone_index = pinecone.Index("sitemap")
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)

        self.context = None
        self.prompt = None
        self.sources = []

        self.context_nodes = self._get_nodes()

    def _get_nodes(self):
        # Get context information relevant to the question
        self.vector_store_query = VectorStoreQuery(
            query_embedding=self.embedding, 
            similarity_top_k=self.top_k,
            mode=self.mode
        )

        query_result = self.vector_store.query(self.vector_store_query)
        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
            self.sources.append(node.metadata['Source'])
        self.sources = list(set(self.sources))
        return nodes_with_scores

    def get_prompt(self):
        # Create text for LLM
        prompt_template = PromptTemplate(
            """\
            Context information is below.
            ---------------------
            {context}
            ---------------------
            Given the context information and not prior knowledge, answer the query.
            Query: {question}
            Answer: \
            """
        )
        self.context = "\n\n".join([n.get_content() for n in self.context_nodes])
        self.prompt = prompt_template.format(context=self.context, question=self.question)
        return self.prompt
