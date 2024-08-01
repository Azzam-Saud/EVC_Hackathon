document.addEventListener('DOMContentLoaded', function() {
    // Function to handle adding items to the cart
    function addToCart(productName, price) {
        fetch('/add_to_cart', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                product: productName,
                price: price,
                quantity: 1 // You can modify this if you want to handle quantity differently
            }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to add item to cart. Server responded with ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            alert(`${data.message} Added: ${productName} to cart.`);
            fetchCartItems(); // Update cart display after adding item
        })
        .catch(error => {
            console.error('Error adding item to cart:', error);
            alert('Failed to add item to cart.');
        });
    }

    // Event delegation for 'Add to Cart' buttons
    document.addEventListener('click', function(event) {
        if (event.target.classList.contains('add-to-cart-btn')) {
            const productName = event.target.getAttribute('data-product');
            const price = parseFloat(event.target.getAttribute('data-price'));
            addToCart(productName, price);
        }
    });

    // Function to fetch cart items from server and display them
    function fetchCartItems() {
        const cartItemsContainer = document.getElementById('cart-items');
        const cartTotalValue = document.getElementById('cart-total-value');

        fetch('/get_cart_items')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to fetch cart items. Server responded with ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                // Clear existing cart items
                cartItemsContainer.innerHTML = '';

                // Iterate over received cart items and display them
                data.items.forEach(item => {
                    const cartItemElement = document.createElement('div');
                    cartItemElement.classList.add('cart-item');
                    cartItemElement.innerHTML = `
                        <div class="cart-item-details">
                            <h4>${item.product}</h4>
                            <p>Price: $${item.price.toFixed(2)}</p>
                            <p>Quantity: ${item.quantity}</p>
                        </div>
                        <button class="remove-from-cart-btn" data-product="${item.product}">Remove</button>
                    `;
                    cartItemsContainer.appendChild(cartItemElement);
                });

                // Update total cart value
                cartTotalValue.textContent = data.total.toFixed(2);
            })
    }

    // Fetch cart items when the page loads
    fetchCartItems();

    // Event delegation for 'Remove' buttons
    document.addEventListener('click', function(event) {
        if (event.target.classList.contains('remove-from-cart-btn')) {
            const productName = event.target.getAttribute('data-product');
            removeFromCart(productName);
        }
    });

    // Function to remove item from cart
    function removeFromCart(productName) {
        fetch(`/remove_from_cart/${productName}`, {
            method: 'DELETE',
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to remove item from cart. Server responded with ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            fetchCartItems(); // Update cart display after removing item
            alert(`${data.message} Removed: ${productName} from cart.`);
        })
        .catch(error => {
            console.error('Error removing item from cart:', error);
            alert('Failed to remove item from cart.');
        });
    }
});
