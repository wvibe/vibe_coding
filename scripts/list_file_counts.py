import os
import argparse
from collections import defaultdict
import sys

def normalize_extensions(ext_string):
    """
    Normalizes a comma-separated string of extensions.

    Args:
        ext_string: A string like "jpg, .png ,TXT".

    Returns:
        A list of normalized extensions, e.g., ['.jpg', '.png', '.txt'],
        or an empty list if ext_string is None or empty.
    """
    if not ext_string:
        return []
    extensions = [ext.strip().lower() for ext in ext_string.split(',')]
    # Ensure all extensions start with a dot
    return [ext if ext.startswith('.') else '.' + ext for ext in extensions if ext]

def count_files(filenames, normalized_extensions):
    """
    Counts total files and files matching specific extensions.

    Args:
        filenames: A list of filenames in a directory.
        normalized_extensions: A list of normalized extensions (e.g., ['.jpg']).

    Returns:
        A tuple: (total_count, extension_counts_dict).
        extension_counts_dict maps each normalized extension to its count.
    """
    total_count = len(filenames)
    extension_counts = defaultdict(int)

    if not normalized_extensions:
        return total_count, {}

    for filename in filenames:
        try:
            _, ext = os.path.splitext(filename)
            ext_lower = ext.lower()
            if ext_lower in normalized_extensions:
                extension_counts[ext_lower] += 1
        except Exception:
            # Ignore files that might cause errors with splitext
            pass

    # Ensure all requested extensions are in the dict, even if count is 0
    for ext in normalized_extensions:
        if ext not in extension_counts:
            extension_counts[ext] = 0

    return total_count, dict(sorted(extension_counts.items()))

def main():
    parser = argparse.ArgumentParser(
        description="List file counts in a directory and its subdirectories."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="The root directory to scan."
    )
    parser.add_argument(
        "--exts",
        help="Comma-separated list of file extensions to count (e.g., jpg,txt,png)."
    )
    parser.add_argument(
        "--path-width",
        type=int,
        default=15,
        help="Width for the path column in the output."
    )
    parser.add_argument(
        "--include-zero",
        action='store_true',
        default=False,
        help="Include directories with zero total files in the output. By default, they are skipped."
    )

    args = parser.parse_args()

    root_dir = args.root
    path_pad = args.path_width

    if not os.path.isdir(root_dir):
        print(f"Error: Root directory not found or is not a directory: {root_dir}", file=sys.stderr)
        sys.exit(1)

    normalized_extensions = normalize_extensions(args.exts)

    # Determine the maximum width needed for the labels ("all" and extensions)
    label_pad = len("all")
    if normalized_extensions:
        max_ext_len = max(len(ext) for ext in normalized_extensions) if normalized_extensions else 0
        label_pad = max(label_pad, max_ext_len)

    # Use os.path.normpath to handle potential trailing slashes nicely
    abs_root = os.path.abspath(os.path.normpath(root_dir))
    root_len = len(os.path.dirname(abs_root)) + 1 # +1 for the separator

    first_dir = True
    for dirpath, _, filenames in os.walk(root_dir, topdown=True):
        if not first_dir:
             print() # Add a blank line between directory outputs
        first_dir = False

        total_count, extension_counts = count_files(filenames, normalized_extensions)

        # Skip directories with zero total files unless --include-zero is specified
        if total_count == 0 and not args.include_zero:
            continue

        # Get relative path for display
        relative_path = os.path.relpath(dirpath, start=root_dir)

        # Print the 'all' count line
        print(f"{relative_path:<{path_pad}} {'all':<{label_pad}} : {total_count} file(s)")

        # Print extension counts if requested
        if normalized_extensions:
            # Pad the start of the extension lines to align labels
            ext_line_prefix = ' ' * path_pad
            for ext, count in extension_counts.items():
                 print(f"{ext_line_prefix}{ext:<{label_pad}} : {count} file(s)")


if __name__ == "__main__":
    main()