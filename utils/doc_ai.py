import re
from typing import Optional
import os
import json

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InternalServerError
from google.api_core.exceptions import RetryError
from google.cloud import documentai  
from google.cloud import storage



def batch_process_documents(
    project_id: str,
    location: str,
    processor_id: str,
    gcs_output_uri: str,
    processor_version_id: Optional[str] = None,
    gcs_input_uri: Optional[str] = None,
    input_mime_type: Optional[str] = None,
    gcs_input_prefix: Optional[str] = None,
    field_mask: Optional[str] = None,
    timeout: int = 400,
    chunk_size=1000,
    
) -> None:
    # define the chunking process option
    process_options = documentai.ProcessOptions(
        layout_config=documentai.ProcessOptions.LayoutConfig(
            chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
                chunk_size=chunk_size,
                include_ancestor_headings=True,
            )
        )
    )
    
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # for a single file
    if gcs_input_uri:
        gcs_document = documentai.GcsDocument(
            gcs_uri=gcs_input_uri, mime_type=input_mime_type
        )
        gcs_documents = documentai.GcsDocuments(documents=[gcs_document])
        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
    # for a bunch of files in the directory
    else:
        gcs_prefix = documentai.GcsPrefix(gcs_uri_prefix=gcs_input_prefix)
        input_config = documentai.BatchDocumentsInputConfig(gcs_prefix=gcs_prefix)

    # Cloud Storage URI for the Output Directory
    gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
        gcs_uri=gcs_output_uri, field_mask=field_mask
    )

    # Where to write results
    output_config = documentai.DocumentOutputConfig(gcs_output_config=gcs_output_config)

    # processor_version_id
    if processor_version_id:
        name = client.processor_version_path(
            project_id, location, processor_id, processor_version_id
        )
    else:
        name = client.processor_path(project_id, location, processor_id)

    
    # define the request
    request = documentai.BatchProcessRequest(
        name=name,
        input_documents=input_config,
        document_output_config=output_config,
        process_options=process_options
    )

    # BatchProcess returns a Long Running Operation (LRO)
    operation = client.batch_process_documents(request)

    
    # wait for the operation to complete
    try:
        print(f"Waiting for operation {operation.operation.name} to complete...")
        operation.result(timeout=timeout)
    except (RetryError, InternalServerError) as e:
        print(e.message)

        
    metadata = documentai.BatchProcessMetadata(operation.metadata)
    if metadata.state != documentai.BatchProcessMetadata.State.SUCCEEDED:
        raise ValueError(f"Batch Process Failed: {metadata.state_message}")
        
    print("success, all processing complete")


    
def load_json_from_bucket(bucket_name:str, prefix:str):
    
    '''
    assuming there is only one json file in this folder
    return that json file as json object
    '''
    
    # Get List of Document Objects from the Output Bucket
    storage_client = storage.Client()
    output_blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    # Document AI may output multiple JSON files per source file
    for blob in output_blobs:
        
        # Document AI should only output JSON files to GCS
        if blob.content_type != "application/json":
            print(
                f"Skipping non-supported file: {blob.name} - Mimetype: {blob.content_type}"
            )
            continue

        if blob.name.endswith(".jsonl"):
            continue
            
        print(f"Fetching {blob.name}")

        # document = documentai.Document.from_json(
        #     blob.download_as_bytes(), ignore_unknown_fields=True
        # )
        # pdf_json = json.loads(documentai.Document.to_json(document))

        document=blob.download_as_bytes()
        pdf_json = json.loads(document)
        
        return pdf_json
    

def convert_json_to_jsonl(pdf_json, fileName, fileNum, bucket_name, output_path):
    
    jsonl_output_file= ""

    for chunk in pdf_json['chunkedDocument']['chunks']:

        jsonl_line=json.dumps({
            "chunkID": chunk['chunkId'],
            "fileName": fileName,
            "fileNum": fileNum,
            "pageStart": chunk['pageSpan']['pageStart'],
            "pageEnd": chunk['pageSpan']['pageEnd'],
            "content": chunk['content']
        })

        if jsonl_line:
            jsonl_output_file += jsonl_line+"\n"

    
    lines=jsonl_output_file.splitlines()
    print('number of lines:', len(lines))
    jsonl_output_file="\n".join(line for line in lines if line.strip())    
    
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(output_path)
    blob.upload_from_string(jsonl_output_file, content_type="application/json")
    
    
def read_jsonl_to_json_list(bucket_name, prefix):
    
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        if blob.content_type != "application/json":
            continue

        if blob.name.endswith(".json"):
            continue
        print(f"current file name: {blob.name}")    


        blob_bytes = blob.download_as_bytes()
        jsonl_content = blob_bytes.decode('utf-8')
        json_lines = jsonl_content.strip().split('\n')

        json_list=[]
        for line in json_lines:
            if line:  # Ensure the line is not empty
                json_object = json.loads(line)
                json_list.append(json_object)
    
    return json_list
    
    
