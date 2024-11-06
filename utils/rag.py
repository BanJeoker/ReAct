import os
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document


class RAG:
    def __init__(self, pdf_path, chunking_method, faiss_vector_store=None, input_chunks=None):
        '''
        pdf_path: path to the PDF files
        chunking_method: name of the chunking method
        '''
        # save a copy of the original text using PyPDF2
        self.text = self.read_pdf(pdf_path)
        
        self.chunks=None
        if input_chunks:
            self.chunks=[Document(page_content=chunk) for chunk in input_chunks]
        else:
            # get the chunk, we do not use "self.text" directly becuase some methods, like the unstrucutre pacakge, starts from reading the file path directly.
            self.chunks=self.get_chunks(pdf_path, chunking_method)
            
        self.embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
        
        # initialize different retriever, including faiss_retriever and bm25_retriever
        if faiss_vector_store is None: # faiss store is slow to create if the embedding model is large, so we load a pre-computed one
            self.faiss_vector_store = FAISS.from_documents(self.chunks, self.embedding_model)
        else:
            self.faiss_vector_store = faiss_vector_store            
        self.faiss_retriever = self.faiss_vector_store.as_retriever(search_kwargs={"k": 3}) # k is the number of returned chunks

        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k =  3 # k is the number of returned chunks

        # create the ensemble retriever, the ensemble retriever will sort all the returned document by order based on its sorting criteria
        self.ensemble_retriever = EnsembleRetriever(retrievers=[self.faiss_retriever,self.bm25_retriever], weights=[0.5, 0.5])
        
    def get_chunks(self, pdf_path, chunking_method):
        '''
        get chunks based on different chunking methods
        '''
        if chunking_method=='unstructured': # the package unstructured
            return self.get_chunks_unstructured(pdf_path)

        elif chunking_method=='recursive':
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
            chunks=text_splitter.create_documents([self.text])
            return chunks
                      
    def get_chunks_unstructured(self, pdf_path):
        '''
        to be implemented
        '''
        return None
    
    def read_pdf(self, pdf_path):
        '''
        read pdf using PyPDF2
        '''
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    

    def search(self, query, method, num_top_chunks=3):
        
        '''
        search function
        query: the input query
        method: the method to be used, 'ensemble' uses the ensemble retriever, 'all' returns the whole pdf
        '''
        if method=='ensemble':
            documents=self.ensemble_retriever.invoke(input=query)
            text=""
            # return the top 3 chunks connected together
            for d in documents[:num_top_chunks]:
                text+=d.page_content
            return text
        
        elif method=='all':
            # returned the whole pdf
            return self.text

