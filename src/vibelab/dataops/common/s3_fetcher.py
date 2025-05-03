"""
Provides functionality to fetch content from S3 URIs.

This module uses boto3 to interact with AWS S3 and relies on standard boto3
credential discovery (environment variables, shared credential file, IAM roles, etc.).
Ensure AWS credentials are configured correctly in the environment where this code runs.
"""

import functools
import logging
from typing import Tuple

import boto3
import botocore

# Configure logging
logger = logging.getLogger(__name__)

# --- Helper functions copied from ref/data_management/resource_manager.py ---
# Source: ref/data_management/resource_manager.py
S3_PREFIX = "s3://"


# Source: ref/data_management/resource_manager.py
def is_s3_uri(uri: str) -> bool:
    """Checks if the given URI string starts with the S3 prefix."""
    return uri.startswith(S3_PREFIX)


# Source: ref/data_management/resource_manager.py
def get_s3_bucket_key(uri: str) -> Tuple[str, str]:
    """Parses an S3 URI to extract the bucket name and object key.

    Args:
        uri: The S3 URI string (e.g., 's3://my-bucket/path/to/object.txt').

    Returns:
        A tuple containing (bucket_name, object_key).

    Raises:
        RuntimeError: If the URI is not a valid S3 URI format.
    """
    if not is_s3_uri(uri):
        # This should ideally be caught by the caller, but added defensively.
        raise RuntimeError("This method should only be called for s3 uri!")
    try:
        # Remove prefix and split by the first '/'
        bucket, key = uri[len(S3_PREFIX) :].split("/", maxsplit=1)
        return bucket, key
    except ValueError as e:
        raise ValueError(f"Invalid S3 URI format: '{uri}'. Could not split into bucket/key.") from e


# --- S3 Fetcher Implementation ---


@functools.lru_cache(maxsize=128)
def fetch_s3_uri(s3_uri: str) -> bytes:
    """Fetches the content of an object from an S3 URI.

    Uses boto3 to download the object's content. Results are cached in memory
    using lru_cache for efficiency within a single process execution.

    Relies on standard boto3 credential discovery mechanisms.

    Args:
        s3_uri: The S3 URI of the object to fetch (e.g., 's3://my-bucket/data.csv').

    Returns:
        The raw byte content of the S3 object.

    Raises:
        ValueError: If the provided s3_uri is not a valid S3 URI format.
        FileNotFoundError: If the specified S3 object (key) does not exist in the bucket.
        IOError: If there's a Boto3 ClientError (e.g., access denied) or other
                 unexpected error during the fetch operation.
    """
    logger.debug("Attempting to fetch S3 URI: %s", s3_uri)

    if not is_s3_uri(s3_uri):
        logger.error("Invalid S3 URI format provided: %s", s3_uri)
        raise ValueError("Invalid S3 URI format. Must start with 's3://'.")

    try:
        bucket_name, object_key = get_s3_bucket_key(s3_uri)
        logger.debug("Parsed Bucket: %s, Key: %s", bucket_name, object_key)

        # Initialize S3 client - credentials handled by boto3's default chain
        s3_client = boto3.client("s3")

        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        content: bytes = response["Body"].read()

        logger.debug("Successfully fetched %d bytes from %s", len(content), s3_uri)
        return content

    except botocore.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        logger.error("Boto3 ClientError fetching %s (Code: %s): %s", s3_uri, error_code, e)
        if error_code == "NoSuchKey":
            raise FileNotFoundError(f"S3 object not found: {s3_uri}")
        else:
            # Covers permission errors, throttling, etc.
            raise IOError(f"Failed to fetch S3 object {s3_uri} due to AWS ClientError: {e}") from e
    except ValueError as e:  # Catch parsing errors from get_s3_bucket_key
        logger.error("Failed to parse S3 URI %s: %s", s3_uri, e)
        raise  # Re-raise the ValueError as it indicates bad input
    except Exception as e:
        # Catch other potential issues (e.g., boto3 setup, network)
        logger.exception("An unexpected error occurred fetching %s: %s", s3_uri, e, exc_info=True)
        raise IOError(f"An unexpected error occurred fetching {s3_uri}: {e}") from e


if __name__ == "__main__":
    import sys

    # Configure basic logging for the test run
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Default public S3 URI for testing (can be overridden by command line arg)
    # Example public file: A small model file from PyTorch hub
    DEFAULT_TEST_URI = "s3://pytorch/models/resnet18-5c106cde.pth"

    # Check if a URI was provided as a command line argument
    if len(sys.argv) > 1:
        test_uri = sys.argv[1]
        logger.info(f"Using provided S3 URI for testing: {test_uri}")
    else:
        test_uri = DEFAULT_TEST_URI
        logger.info(f"Using default public S3 URI for testing: {test_uri}")
        logger.info("You can provide a specific S3 URI as a command line argument.")

    logger.info(f"Attempting to fetch: {test_uri}")
    try:
        # Attempt to fetch the content
        content = fetch_s3_uri(test_uri)
        logger.info(
            f"Successfully fetched {len(content)} bytes from {test_uri}. AWS credentials seem okay!"
        )
        # Optional: Uncomment to save the first few bytes to a file for inspection
        # with open("s3_fetch_test_output.bin", "wb") as f:
        #     f.write(content[:1024]) # Write first 1KB
        # logger.info("Saved first 1KB of content to s3_fetch_test_output.bin")

    except FileNotFoundError as e:
        logger.error(f"Error: File not found on S3. {e}")
    except ValueError as e:
        logger.error(f"Error: Invalid S3 URI format. {e}")
    except IOError as e:
        logger.error(
            f"Error: Could not fetch from S3. Check AWS credentials and permissions. Details: {e}"
        )
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the test: {e}", exc_info=True)
