import dataProcessor
import matplotlib.pyplot as plt
import matcher

# Get data from data folder (which is unavalible on github)
image_paths=dataProcessor.search_data()
images=dataProcessor.reader(image_paths,[0,9])

# Create matcher instance object
matcherInst = matcher.Matcher((1098, 1098))

# Calculate matching points of two images
corresp = matcherInst(images[0], images[1], 0.9, False)

# Draw acquired matches
a=matcher.Matcher.draw_matches(corresp)
plt.show()

