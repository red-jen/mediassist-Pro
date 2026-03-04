from langchain_core.documents import Document

doc = Document(page_content="hello", metadata={"page": 1})
print("Document:", doc)
print("Document page_content:", doc.page_content)
print("Document metadata:", doc.metadata)

doc_dict = {"page_content": "hello", "metadata": {"page": 1}}
print("Dict:", doc_dict)
print("Dict page_content:", doc_dict["page_content"])
print("Dict metadata:", doc_dict["metadata"])