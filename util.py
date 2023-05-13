import io
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from steamship import Block, Steamship


def show_ouput(output):
    """Display the multi-modal output from the agent."""
    UUID_PATTERN = re.compile(
        r"([0-9A-Za-z]{8}-[0-9A-Za-z]{4}-[0-9A-Za-z]{4}-[0-9A-Za-z]{4}-[0-9A-Za-z]{12})"
    )

    outputs = UUID_PATTERN.split(output)
    # Clean trailing and leading non-word characters
    outputs = [re.sub(r"^\W+", "", el) for el in outputs]
    out_uuid = outputs[1]

    image_file = io.BytesIO(Block.get(Steamship(), _id=out_uuid).raw())

    image = mpimg.imread(image_file)

    plt.imshow(image)
    plt.axis('off')
    plt.show()
