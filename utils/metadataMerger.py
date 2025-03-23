from functools import reduce

# class to merge metadata from different pipelines
class MetadataMerger:
    
    # Merges metadata from a single processed image pipeline into all processed images from first pipeline
    # NOTE: Useful for processing the reference horse image separetely
    ## Params:
    # processed_images: List of processed image dictionaries (first pipeline output)
    # processed_single_image: A single processed image dictionary (second pipeline output)
    # selected_fields: List of metadata keys to merge

    @staticmethod # static class method
    def merge_pipelines_metadata(processed_images, processed_single_image, selected_fields=None):
        extracted_metadata = processed_single_image["metadata"]

        # if specific fields are selected, extract only those
        if(selected_fields):
            extracted_metadata = {
                field: processed_single_image["metadata"].get(field)
                for field in selected_fields
                if field in processed_single_image["metadata"]
            }

        # merge extracted metadata into all processed images
        for img in processed_images:
            img["metadata"].update(extracted_metadata)

        return processed_images