You resize one storyboard ad into banner ads of these sizes: $SIZES, for the brand $BRAND.
Return every size in one answer. For each, return HTML with one <style> element and one <div id="ad">.
Inside #ad, put one <div class="frame" data-scene="ID" data-duration="MS"> per frame, in order, as direct children,
ending with an empty data-scene="end" frame.
Give text elements data-role and data-el (the element id); wrap bold words in <strong data-role="highlight">.
Product images: <img src="asset://ID"> only.
Don't write scripts, animation, click handling or the logo; code adds those.
Follow the brand kit's colours, font, typography limits, logo position and reserved areas, and keep all copy exactly
as given.
