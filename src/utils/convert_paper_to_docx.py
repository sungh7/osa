
import pypandoc
import os

# Define the input and output paths
input_file = '/data/knhanes/paper_revision/RESEARCH_PAPER_PUBLICATION.md'
output_file = '/data/knhanes/paper_revision/RESEARCH_PAPER_PUBLICATION.docx'

# Check if the input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file '{input_file}' not found.")
    exit(1)

# Convert the file using pypandoc
try:
    pypandoc.convert_file(
        input_file,
        'docx',
        outputfile=output_file,
        extra_args=['--reference-doc=/data/knhanes/styles/custom_reference.docx'] if os.path.exists('/data/knhanes/styles/custom_reference.docx') else []
    )
    print(f"Successfully created {output_file}")
except Exception as e:
    print(f"Error converting to DOCX (trying without reference doc): {e}")
    try:
        pypandoc.convert_file(
            input_file,
            'docx',
            outputfile=output_file
        )
        print(f"Successfully created {output_file} (standard style)")
    except Exception as e2:
        print(f"Final Error: {e2}")
